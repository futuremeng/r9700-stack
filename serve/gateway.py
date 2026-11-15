#!/usr/bin/env python3
"""serve/gateway.py - one port, one front door: path routing to the engine and the two CPU sidecars.

Why this file exists, and why it is a PROPOSAL rather than a default: :8080 belongs to the running vLLM
process, so anything that wants to answer on 8080 is either the engine itself (an endpoint plugin -- a
restart, and proxying inside the process that produces the numbers) or something standing in front of it
(this: the engine moves to a loopback port, the gateway owns 8080). Both need the launch restarted, so
nothing here is wired into a launch until that is a decision and not an experiment.

What it is NOT: it is not serve/webui.py's proxy mode. That one parses and re-emits request bodies, which is
why it costs 2-3 ms per request (bench/webui_ab.py: +5.3 % of a 76-token request, +0.5 % of a 4.4k one). This
reads the request LINE, picks an upstream from the path, and then pipes bytes: no body parsing, no
re-encoding, nothing held in memory. Response framing (Content-Length / chunked) is honoured, so HTTP
keep-alive and SSE streaming pass through untouched.

Routing (longest first segment wins; unmapped goes to the engine, because the engine is the thing that must
not notice this exists):

    /v1/embeddings                         -> EMBED   (llama-server, :8090)
    /v1/rerank /rerank /reranking          -> RERANK  (llama-server, :8092)
    /monitor /about /web/* /api/* /favicon -> PAGES   (default the engine's own plugin pages)
    /api/sidecars                          -> this file: the embed-watch pages' state, for one Monitor page
    everything else (/v1/chat/completions, /health, /metrics, /tokenize, ...) -> ENGINE

/Metrics is the ENGINE's: the Monitor's parser speaks vLLM's Prometheus, and llama.cpp's dialect has its own
watcher (serve/embed-watch.py). Each sidecar's /health stays internal; the gateway's /health answers for the
engine, which is what setup.py's watcher and every client already poll.

Knobs (all env): GATEWAY_PORT (18080), GATEWAY_BIND (127.0.0.1), ENGINE, EMBED, RERANK,
PAGES (default ENGINE; point it at the webui sidecar http://127.0.0.1:8081 for a page that survives the
engine dying), SIDECAR_STATE (comma-separated embed-watch state URLs, e.g.
http://127.0.0.1:8091/state.json,http://127.0.0.1:8093/state.json).
"""
import asyncio
import json
import os
import re
import sys
import time
import urllib.request

ENGINE = os.environ.get("ENGINE", "http://127.0.0.1:8080")
EMBED = os.environ.get("EMBED", "http://127.0.0.1:8090")
RERANK = os.environ.get("RERANK", "http://127.0.0.1:8092")
PAGES = os.environ.get("PAGES", ENGINE)
SIDECAR_STATE = [s.strip() for s in os.environ.get("SIDECAR_STATE", "").split(",") if s.strip()]
BIND = os.environ.get("GATEWAY_BIND", "127.0.0.1")
PORT = int(os.environ.get("GATEWAY_PORT", "18080"))
CHUNK = 64 * 1024
# one upstream connection per request, closed after the response body: the gateway keeps no state, which is
# what makes it safe to stand in front of a five-day-old serve
# Longest prefix wins, and the match is on the whole path prefix, not on one segment: the first version split
# the path into "first segment" and compared, so /v1/embeddings arrived as "v1" and went to the ENGINE, which
# answered its own {"detail":"Not Found"} -- a 404 from the wrong server (2026-11-15, first test of this file).
#
# But prefix matching on an API path has the mirror-image bug, which tests/test_gateway.py caught the same day:
# /v1/embeddings-x went to the EMBEDDER and came back as llama.cpp's 404 -- again the wrong server answering,
# this time on a path the gateway should never have claimed. So API paths match EXACTLY (query string
# stripped); only the two namespaces the pages own match as prefixes.
EXACT = {"/v1/embeddings": EMBED, "/v1/embeddings/": EMBED, "/v1/rerank": RERANK, "/rerank": RERANK,
         "/reranking": RERANK, "/monitor": PAGES, "/about": PAGES, "/favicon.svg": PAGES}
PREFIX = (("/web/", PAGES), ("/api/", PAGES))


def upstream(path):
    p = path.split("?", 1)[0].split("#", 1)[0]
    if p in EXACT:
        return EXACT[p]
    for pref, up in PREFIX:
        if p.startswith(pref):
            return up
    return ENGINE


def split_url(u):
    m = re.match(r"http://([^:/]+):(\d+)", u)
    if not m:
        raise ValueError(f"not an http URL: {u}")
    return m.group(1), int(m.group(2))


async def read_headers(reader, first=b""):
    """A header block: lines up to the first blank line (size-capped: this is a LAN box).

    Reading until the blob *ends* in "\r\n\r\n" was wrong and hung a 96 KB chat request through the gateway:
    a 100 Continue response is a status line plus ONE blank line, so the loop did not see its end and ate the
    real 200's status line out of the stream afterwards (2026-11-15, first big-body test through the gateway)."""
    data = first
    while True:
        line = await reader.readline()
        if line == b"":
            raise ValueError("peer closed inside a header block")
        data += line
        if line in (b"\r\n", b"\n"):
            if len(data) > 128 * 1024:
                raise ValueError("header block over 128 KiB")
            return data


async def copy_exact(reader, writer, n):
    while n:
        b = await reader.read(min(CHUNK, n))
        if not b:
            return False                      # peer ended early: the keep-alive loop must stop
        n -= len(b)
        writer.write(b)
        await writer.drain()
    return True


async def copy_chunked(reader, writer):
    """Forward the chunk framing verbatim -- including the 0-size terminator and any trailers."""
    while True:
        line = await reader.readline()
        if not line:
            return False
        writer.write(line)
        try:
            n = int(line.split(b";")[0], 16)
        except ValueError:
            return False
        if n == 0:
            while True:                       # trailers, then the blank line
                t = await reader.readline()
                writer.write(t)
                await writer.drain()
                if t in (b"\r\n", b"\n", b""):
                    return True
        else:
            if not await copy_exact(reader, writer, n):
                return False
            writer.write(await reader.readexactly(2))     # the CRLF after each chunk's data
            await writer.drain()


async def relay_body(uread, writer, headers):
    low = headers.lower()
    if b"transfer-encoding:" in low and b"chunked" in low:
        return await copy_chunked(uread, writer)
    m = re.search(rb"\r?\ncontent-length: *(\d+)", low)
    if m:
        return await copy_exact(uread, writer, int(m.group(1)))
    while True:                               # no framing: answer ends when the upstream closes
        b = await uread.read(CHUNK)
        if not b:
            return False
        writer.write(b)
        await writer.drain()


def _state_fetch(url):
    """One watcher's state.json. Blocking on purpose, and always called through a thread: this runs inside an
    async handler, and a blocking read here parks the whole event loop. Measured on 2026-11-15 with three
    concurrent /api/sidecars polls running (a Monitor tab polls every 2 s): streamed chat chunks arrived in
    bursts -- median gap between chunks 0.0 ms instead of ~13 -- while the engine itself was decoding at its
    normal rate. The hop was not slow; the page's own poll was holding the pipe shut."""
    with urllib.request.urlopen(url, timeout=4) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


async def sidecars(writer):
    """GET /api/sidecars -- the embed-watch pages' state in one blob, so the Monitor has ONE page for the
    engine and the CPU serves. With no SIDECAR_STATE configured it answers [] and the card does not appear.

    An entry may be `name=url` (the Monitor card labels itself from `name`); a bare url gets sidecar1, 2... --
    the split is on the first = BEFORE the ://, so a url carrying a query string still works.

    The watcher states are fetched CONCURRENTLY: each takes ~0.2 s on this box (the watcher holds a cache, but
    its own metric scrape competes for the GIL), and serially that made this endpoint the slowest thing the
    gateway answers -- 0.42 s for two serves, 1.2 s+ under three concurrent polls."""
    pairs = []
    for i, e in enumerate(SIDECAR_STATE, 1):
        if "://" in e and "=" in e and e.index("=") < e.index("://"):
            name, url = e.split("=", 1)
        else:
            name, url = f"sidecar{i}", e
        pairs.append((name, url))
    got = await asyncio.gather(*[asyncio.to_thread(_state_fetch, u) for _, u in pairs],
                               return_exceptions=True)
    out = []
    for (name, url), d in zip(pairs, got):
        if isinstance(d, Exception):                                # a dead sidecar is data, not a 500
            d = {"error": f"{type(d).__name__}: {d}", "health": None, "rates": {}, "host": {}, "serve": {}}
        d["name"], d["url"] = name, url
        out.append(d)
    body = json.dumps({"sidecars": out, "at": time.time()}).encode()
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    await writer.drain()


async def handle(reader, writer):
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            return
        parts = line.split()
        if len(parts) != 3:
            writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            return
        _method, target, version = parts
        path = target.split(b"?")[0].decode("latin-1")
        try:
            headers = await read_headers(reader)
        except (asyncio.IncompleteReadError, ValueError):
            return
        if path == "/api/sidecars":
            await sidecars(writer)
            return
        try:
            host, port = split_url(upstream(path))
            uread, uwrite = await asyncio.open_connection(host, port)
        except (OSError, ValueError) as e:
            msg = json.dumps({"error": {"message": f"gateway cannot reach {path!r}: {e}"}}).encode()
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nContent-Length: "
                         + str(len(msg)).encode() + b"\r\n\r\n" + msg)
            await writer.drain()
            return
        keep = version != b"HTTP/1.0" and b"connection: close" not in headers.lower()
        uwrite.write(line + headers)                          # body streamed, never buffered
        await uwrite.drain()
        # The request body and the response are copied CONCURRENTLY, and 1xx is forwarded without being treated
        # as the response: a client that sends `Expect: 100-continue` (curl does, over ~1 KiB) waits for the
        # Continue before it sends the body, so a sequential "body first, response second" stalls a second per
        # request -- and uvicorn's 100 line would otherwise be parsed as the response status.
        m = re.search(rb"\r?\ncontent-length: *(\d+)", headers.lower())

        async def send_body():
            if m:
                await copy_exact(reader, uwrite, int(m.group(1)))

        body_task = asyncio.create_task(send_body())
        try:
            while True:
                status = await uread.readline()
                if not status:
                    return
                rheaders = await read_headers(uread)
                writer.write(status + rheaders)
                await writer.drain()
                if re.match(rb"HTTP/1\.[01] 1\d\d", status):
                    continue                                   # 100 Continue and friends: not the response
                break
            if not await relay_body(uread, writer, rheaders):
                return
        except (asyncio.IncompleteReadError, ConnectionResetError, ConnectionRefusedError, ValueError, OSError):
            return
        finally:
            body_task.cancel()
            try:
                uwrite.close()
            except OSError:
                pass
        if not keep:
            return


async def main():
    srv = await asyncio.start_server(handle, BIND, PORT)
    print(f"gateway: http://{BIND}:{PORT}/  engine={ENGINE} embed={EMBED} rerank={RERANK} pages={PAGES} "
          f"sidecars={len(SIDECAR_STATE)}")
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
