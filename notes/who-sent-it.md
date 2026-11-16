# Who sent it: what a LiteLLM proxy lets the box see, measured (2026-11-16)

Asked whether the Monitor could attribute a 125k cold context to whoever ran it, the reflex answer — source IP — is
worthless here: every request arrives from LiteLLM. So the question became: **what survives the proxy?** That is
answerable without guessing, and the answer is version-specific enough to be worth writing down with the version
attached.

## How it was measured

`tools/litellm-spy.py` listens on `:4010`, records the headers and **body key names** of everything a proxy sends
it, and forwards the body verbatim to the front door on `:8080` (so the answer is a real answer, not an echo).
LiteLLM 1.109.1 with the `openai` provider (`openai` 3.51.0), installed under `/tmp/litest` — not into either
venv. Two probes through a proxy whose `api_base` pointed at the spy:

- **A** — model `Qwen3.8`; the client sent `X-R9700-Tag: client-tag`, `X-Session-Id: sess-77`, a custom
  `User-Agent`, and the OpenAI body field `user: "alice-probe"`.
- **B** — model `Qwen3.8-dev`, whose `litellm_params` carried `extra_headers: {"X-R9700-Tag": "yaml-dev"}`; the
  client sent nothing custom.

## What arrives at the front door

| what the client or config tried to send | arrives? |
|---|---|
| client's custom header (`X-R9700-Tag`, `X-Session-Id`) | **no** — dropped entirely |
| client's `User-Agent` | **no** — replaced by `AsyncOpenAI/Python 3.51.0` |
| body field `user` | **no** — the arriving body's keys were `max_tokens, messages, model, temperature` |
| the model alias the client asked for (`Qwen3.8-dev`) | **no** — upstream body said `model: Qwen3.8` |
| the client's bearer token | **no** — upstream sees the deployment's key, `authorization: Bearer none` |
| `litellm_params.extra_headers` | **yes** — `x-r9700-tag: yaml-dev` arrived verbatim |
| litellm's own fingerprint | **yes** — `x-stainless-*`: package version 3.51.0, `x-stainless-read-timeout: 6000.0`, `x-stainless-retry-count: 0` |

And on the engine side there is nowhere for a body-level identity to go even if it survived: `vllm/v1/metrics/stats.py`
in 0.32.0 has no user field in the finished-request records, and those records are what the Monitor's request rows
are built from on this box. So attribution **has to happen at the front door, on a header the proxy was told to
send** — not on an IP, not on the `user` field, not on the model name.

## The allocation that actually works

One deployment entry per *kind* of caller, differing only by the tag header. The upstream model name stays
`Qwen3.8` (that is the served name, and vLLM checks it), which is precisely why the header is the only handle:

```yaml
model_list:
  - model_name: Qwen3.8-dev            # interactive: one big cold context, latency matters
    litellm_params:
      model: openai/Qwen3.8
      api_base: http://192.168.1.130:8080/v1
      api_key: none
      extra_headers: {"X-R9700-Tag": "dev"}
  - model_name: Qwen3.8-batch          # data processing: many small requests, throughput matters
    litellm_params:
      model: openai/Qwen3.8
      api_base: http://192.168.1.130:8080/v1
      api_key: none
      extra_headers: {"X-R9700-Tag": "batch"}
```

Callers change a model name, not their code. What the box would then do with it (designed, **not built**): the
gateway copies an allowlisted header — `GATE_TAG_HEADERS=X-R9700-Tag`, one header it has already received, value
clamped to 48 characters and stripped of control characters — onto the request row as `tag`, which rides the
journal like the other per-request facts, and the context-size card grows a per-tag breakdown (rows, cold tokens,
p95 uncached, biggest cold context). Still no body parsing: the gateway's byte-pipe rule is the reason the tag has
to be a header, and it is why the `user` field is a dead end rather than a preference.

## Two facts the probe threw in for free

`x-stainless-retry-count` **does** arrive, and it is honest about client retries. A retried 200k-token prompt is a
second 200k prefill if its blocks have been evicted in between, which is exactly the cost the context-size card
would otherwise under-report — and it is the cost that makes "the same request twice" look free in a client's logs
and expensive in the engine's.

`x-stainless-read-timeout: 6000.0` says when the client will hang up. Measured cold prefills on this box run 5 s at
41k tokens and 26 s at 187k, so a 6000 s timeout never interrupts them; a client configured for 30 s would abandon
precisely the requests the card flags as expensive, then retry them — which is how a client's impatience turns into
doubled prefill work.
