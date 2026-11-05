// serve/web/about.js -- the About page. One read of /api/status and one of /api/host; no polling
// loop beyond a slow refresh, because nothing here is a live curve (that is the Monitor's job).
"use strict";
let status = null;

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
  })[c]);
}

// facts(el, [[label, value, copy?]]) -- a dash row is dropped, a copy row gets a copy button.
function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] !== null && r[1] !== undefined && r[1] !== "")
    .map(([k, v, copy]) => `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code>`
      + ` <button class="copy" data-copy="${esc(v)}" aria-label="Copy ${esc(k)}">copy</button>` : esc(v)}</dd>`)
    .join("");
}

const yesno = (v) => (v === undefined || v === null || v === "" ? null : (String(v) === "false" ? "off" : "on"));
// "Is the prefix cache in effect, and at what granularity?" Read from the engine's own cache_config labels, so the
// answer cannot drift from what the engine runs. A hit can only land on `hash_block_size` (a.k.a. prefix_match_unit)
// when that is set, and otherwise only on a whole physical block -- 832 tokens on the 2026-11-04 cache-on launch,
// measured: a 538-token repeated prompt hit nothing at all, a 3,658-token one hit 3 blocks (2,496 tokens). The
// junction checkpoint is the fork's own knob for resuming an MTP sibling at a shared prefix; it does nothing unless
// the match unit is finer than the mamba block, so it is shown only when it is on.
const cachestate = (c) => {
  if (c.enable_prefix_caching === undefined || c.enable_prefix_caching === null) return null;
  if (String(c.enable_prefix_caching).toLowerCase() !== "true") return "off";
  const none = (v) => v === undefined || v === null || String(v).toLowerCase() === "none";
  const unit = none(c.hash_block_size) ? c.block_size : c.hash_block_size;
  return `on · a hit lands on a ${unit}-token boundary`
    + (String(c.enable_mamba_shared_prefix_checkpoint).toLowerCase() === "true" ? " · junction checkpoint on" : "");
};
const num = (v) => (v === undefined || v === null || v === "" ? null : fmt(Number(v)));

function uptime(s) {
  if (!isnum(s)) return null;
  const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60);
  return (d ? `${d} d ` : "") + (h ? `${h} h ` : "") + `${m} min`;
}

function render() {
  const s = status || {};
  const c = s.config || {}, meta = s.meta || {}, models = s.models || [];
  const model = s.model_name || models[0]?.id || null;
  const host = s.host || {};
  brandStack(s);
  text("mode", s.mode === "proxy" ? "recording requests"
    : s.mode === "merged" ? "inside this serve (serve/plugin)" : "observing (read-only)");
  // "this page is a sidecar" is not true of the merged page, and a page that describes itself wrongly is
  // worse than one that describes nothing.
  if (s.mode === "merged")
    $("api-note").textContent = "this page IS the serve: it was mounted by serve/plugin on this same port, "
      + "it does not relay, and it never sees a request body.";

  facts($("facts-engine"), [
    ["Model", model],
    ["Engine", s.engine?.vllm_version || c.version || null],
    ["Stack", s.stack?.plugin ? `r9700_vllm ${s.stack.plugin}`
      + (s.stack.commit ? ` · ${s.stack.commit}${s.stack.dirty ? " (dirty)" : ""}` : "") : null],
    ["Max context", num(c.max_model_len ?? models[0]?.max_model_len) ? `${fmt(c.max_model_len ?? models[0].max_model_len)} tokens` : null],
    ["Parallelism", c.tensor_parallel_size ? `TP ${c.tensor_parallel_size}`
      + (c.pipeline_parallel_size && c.pipeline_parallel_size !== "1" ? ` · PP ${c.pipeline_parallel_size}` : "")
      + (c.data_parallel_size && c.data_parallel_size !== "1" ? ` · DP ${c.data_parallel_size}` : "") : null],
    ["KV pool", isnum(s.live?.kv_tokens) ? `${fmt(s.live.kv_tokens)} tokens` : null],
    ["KV block", c.block_size ? `${c.block_size} tokens/block` : null],
    ["KV dtype", c.kv_cache_dtype || null],
    ["Prefix caching", cachestate(c)],
    ["Scheduler", c.scheduler_policy || null],
    ["Reachable", s.reachable ? `yes · /health ${s.health ?? 200}` : `no: ${s.error || "not answering"}`],
  ]);

  const keys = Object.keys(meta).sort();
  $("launch-card").hidden = !keys.length;
  if (keys.length) {
    text("launch-src", s.meta_src || "--meta");
    facts($("facts-launch"), keys.map((k) => [k, Array.isArray(meta[k]) ? meta[k].join(", ")
      : typeof meta[k] === "object" ? JSON.stringify(meta[k]) : String(meta[k])]));
  }

  // Configuration and environment, as THREE lists that are allowed to disagree, each naming where it came from:
  // the recipe file is what the launcher recorded for the engine, the engine's labels are what the engine
  // resolved, and the environment / machine-local file are what these processes actually hold. Merging them into
  // one list would hide the disagreement, and `tools/recipe-check.py` exists to turn that disagreement into an
  // exit code. A sidecar's `MAXLEN` proves nothing about the engine, which is why the process rows say which
  // process they are. Nothing on this card defines anything: it reports.
  const env = s.env || {}, efile = env.file || {}, all = c._all || {};
  // A name whose value looks like a credential is listed with its value withheld, so "not printed" is never read
  // as "not set". The payload marks those with a leading ~; the page says it in words instead.
  const envrow = ([k, v]) => (k.startsWith("~")
    ? [k.slice(1), "set · value withheld", null]
    : [k, v, 1]);
  const envRows = Object.entries(env.vars || {}).map(envrow);
  const fileRows = Object.entries(efile.vars || {}).map(envrow);
  facts($("facts-config"), [
    ["This process", env.pid ? `${env.mode === "merged" ? "inside the serve" : "sidecar process"}`
      + ` · pid ${env.pid}` : null],
    ["Its command line", s.argv || null, 1],
    ["Listening on", s.bind ? `${s.bind}:${s.port ?? "?"}` : null],
    ["Machine-local env file", efile.path || null, 1],
    ["The three lists below", "recipe = what the launcher asked for · engine config = what the engine resolved · "
      + "env and file = what the processes hold. They can disagree, and recipe-check is the check that does."],
  ]);
  $("cfg-all-n").textContent = `(${Object.keys(all).length} labels)`;
  facts($("facts-cfgall"), Object.entries(all).sort().map(([k, v]) => [k, String(v), 1]));
  $("env-proc-n").textContent = `(${envRows.length} whitelisted names set)`;
  facts($("facts-env"), envRows);
  $("env-file-n").textContent = `(${fileRows.length} lines)`;
  facts($("facts-envfile"), fileRows);
  text("env-file-note", !env.pid && !efile.path
    // "No env block in the payload" and "no env file on disk" are different sentences, and the second one is what
    // the page used to say about the first: an in-process plugin started before this card existed serves a payload
    // with no env block at all, and "there is no file at undefined" sent someone looking for a file that was never
    // missing. The fix is to name which case it is, and to say what makes the difference -- a restart.
    ? "This page's payload carries no environment block, so this build predates the card: an in-process plugin "
      + "started before it existed. Restart the serve to get one (the files themselves are unchanged)."
    : efile.exists
    ? "The launchers source this file with `set -a` before their own defaults, so these reach the child processes "
      + "and not only the script. The repo keeps its defaults in the repo; what lives here is a fact about this "
      + "box (notes/launch-methodology.md section 10)."
    : `There is no machine-local env file at ${efile.path}, so every default in the repo is in force. `
      + (efile.error ? `(${efile.error})` : ""));

  // Image input gets its own card because the engine cannot testify to it: none of the labels the engine publishes
  // say whether the serve came up --language-model-only. Two witnesses remain, printed apart on purpose: the live
  // process's argv (what the running serve actually accepts) and the recipe file (what this box's launcher recorded
  // for the launch). A serve started by a test config and a recipe file from another tree is exactly the
  // disagreement the Configuration card above refuses to merge.
  const argv = s.argv || "";
  const argvTok = (f) => {            // a flag's value in a command line, JSON values taken whole
    const i = argv.indexOf(f);
    if (i < 0) return null;
    let j = i + f.length;
    while (j < argv.length && argv[j] === " ") j++;
    if (argv[j] === "{") {
      let d = 0, k = j;
      for (; k < argv.length; k++) {
        if (argv[k] === "{") d++;
        else if (argv[k] === "}") { d--; if (!d) { k++; break; } }
      }
      return argv.slice(j, k);
    }
    const m = argv.slice(j).match(/^\S+/);
    return m ? m[0] : null;
  };
  const jsonof = (t) => { try { return t ? JSON.parse(t) : null; } catch (e) { return null; } };
  const lim = jsonof(argvTok("--limit-mm-per-prompt")), mmk = jsonof(argvTok("--mm-processor-kwargs"));
  const mmLabels = Object.keys(all).filter((k) => /(^|_)(mm|vision)(_|$)|modal|image|video/i.test(k)).length;
  const live = !argv
    ? "not in this payload: an in-process page older than the argv line is answering. Restart the serve to get one."
    : argv.includes("--language-model-only")
      ? "no images, no video (--language-model-only)"
      : lim
        ? `${lim.image ?? "?"} image(s), ${lim.video ?? "?"} video(s) per prompt`
          // A limit on the COUNT is not a limit on the COST: the checkpoint's own size default is
          // longest_edge 16,777,216 px = 16,384 image tokens (patch 16 x merge 2 = 1,024 px per token), which is
          // how one screenshot arrived at 15,664 prompt tokens. So "no cap" is said out loud rather than assumed.
          + (mmk && mmk.size ? ` · one image up to ${num(mmk.size.longest_edge)} px (~${
              num(Math.round(Number(mmk.size.longest_edge) / 1024))} tokens)`
            : " · no size cap passed: the checkpoint's own default is 16,777,216 px (~16,384 tokens an image)")
        : "images allowed; the limit is not in the command line this page can see";
  $("mm-card").hidden = !argv && !meta.mm_effective;
  // A sidecar holds its own command line, not the serve's, so on a sidecar page the argv row cannot answer what the
  // engine accepts -- same rule the Configuration card states (a sidecar's MAXLEN proves nothing about the engine).
  facts($("facts-mm"), [
    [s.mode === "merged" ? "The running serve accepts" : "Whose command line this is",
      s.mode === "merged" ? live
        : `the sidecar's, pid ${(s.env || {}).pid || "?"} — it proves nothing about what the serve accepts. `
          + "Open the merged page on the serve's own port for the serve's argv; the recipe row below is shared."],
    ["The recipe file records", meta.mm_effective
      ? `${meta.mm_effective}` + (meta.mm_image === undefined ? ""
        : ` · image ${meta.mm_image}, video ${meta.mm_video}`
          + (Number(meta.mm_max_pix) > 0
            ? ` · one image up to ${num(meta.mm_max_pix)} px (~${num(meta.mm_max_tok)} tokens), floor `
              + `${num(meta.mm_min_pix)} px`
            : " · no image can be sent at all")) : null],
    ["The knobs that choose it", meta.mm_if_on
      ? `MM=${meta.mm ?? "0"} · ${meta.mm_if_on} · MM=0 is this tree's default` : null],
    // Not payload-derived, and labelled as what it is: the arithmetic that turns pixels into prompt tokens comes
    // from this checkpoint's preprocessor_config.json (patch_size 16, merge_size 2).
    ["px per image token", "1,024 (patch 16 x merge 2 = 32 px a side) — from this checkpoint's preprocessor_config.json"],
    ["What the engine's config says", `${mmLabels} label(s) in the config block mention multimodal: the engine is `
      + "silent about this, so it is a flag, and recipe-check comparing the launcher against the live argv is the "
      + "check that can see it"],
  ]);
  text("mm-note", "Measured on this box (notes/multimodal-input.md): one 2,560 x 6,236 screenshot arrived as "
    + "15,664 prompt tokens uncapped and 4,123 with the size cap; drawn text down to 28 px in a canvas that size "
    + "still read correctly at the cap; per-request mm_processor_kwargs are refused, so the cap is a ceiling a "
    + "client cannot lift. Not measured: images under concurrency (every probe so far was one request at a time on "
    + "an idle box), photographs, video.");

  facts($("facts-host"), [
    ["Hostname", host.hostname],
    ["CPU", host.cpu ? `${host.cpu}${host.threads ? `, ${host.threads} threads` : ""}` : null],
    ["RAM", isnum(host.ram) ? `${gb(host.ram, 0)} GiB` : null],
    ["Kernel", host.kernel],
    ["Up", uptime(host.uptime_s)],
  ]);

  const gpus = s.gpus || [];
  $("gpus").replaceChildren(...gpus.map((g) => gpuCard(g, s.gpus_history || [])));

  // A pasted config has to be pasteable on the machine you are sitting at. `upstream` is what this sidecar
  // dials -- 127.0.0.1 on the box that runs the serve -- which is dead wrong pasted into a client anywhere
  // else (this box is reached over the network, and wtmp says so). location.origin is the address that
  // actually answered this page, tunnel or reverse proxy included.
  const origin = (location.origin || "").replace(/\/$/, "");
  const apiBase = `${origin}/v1`;          // LiteLLM and the OpenAI SDK want the /v1; curl does not
  const name = model || "model";
  const rsn = meta.reasoning || null;      // the launcher records the parser it passed, in the recipe file
  facts($("facts-api"), [
    ["Model name", model, true],
    ["OpenAI / LiteLLM base_url", apiBase, true],
    ["Root (curl, /health, /metrics)", origin, true],
    ["Reasoning", rsn ? `separate (${rsn}) — it arrives in message.reasoning_content, and content can come back empty if max_tokens is all spent on it`
      : "not recorded by the launcher"],
    ["API key", s.key_needed ? "the serve has one -- put yours in the environment; it is never printed here"
      : "none: use api_key: none"],
    ["Through this page", s.mode === "proxy" ? `${origin} (point clients here to record them)`
      : s.mode === "merged" ? `${origin} (this IS the serve)` : "not routed here — use the base_url above"],
    ["Weights live at", models[0]?.root ? `${models[0].root} — not a name a client can ask for` : null],
  ]);
  const curl = `curl ${origin}/v1/chat/completions \\\n`
    + `  -H 'Content-Type: application/json'${s.key_needed ? ` \\\n  -H 'Authorization: Bearer $KEY'` : ""} \\\n`
    + `  -d '{"model": "${name}", "messages": [{"role": "user", "content": "hello"}],\n`
    + `       "max_tokens": 64, "temperature": 0}'`;
  $("curl").textContent = curl;
  document.querySelector("[data-copy-target=curl]").dataset.copy = curl;
  // LiteLLM in both shapes people actually use it: a proxy config, and a direct call. The `openai/` prefix is
  // what makes it treat this as an OpenAI-compatible endpoint; the key is never inlined, only named.
  const yaml = `# litellm --config litellm.yaml --port 4000\nmodel_list:\n`
    + `  - model_name: ${name}\n    litellm_params:\n`
    + `      model: openai/${name}\n      api_base: ${apiBase}\n`
    + `      api_key: ${s.key_needed ? "os.environ/LITELLM_API_KEY" : "none"}\n`
    + `      # max_tokens counts reasoning too: this serve reports reasoning_tokens separately.`;
  const py = `import litellm\nr = litellm.completion(\n    model="openai/${name}",\n`
    + `    api_base="${apiBase}",\n    api_key=${s.key_needed ? "os.environ[\"LITELLM_API_KEY\"]" : "\"none\""},\n`
    + `    messages=[{\"role\": \"user\", \"content\": \"hello\"}], max_tokens=128)\n`
    + `m = r.choices[0].message\n`
    + `print(m.content)                              # None when max_tokens was all spent on reasoning\n`
    + `print(getattr(m, "reasoning_content", None))  # where ${rsn ? `this serve's --reasoning-parser ${rsn}` : "a reasoning parser"} puts it`;
  for (const [id, text_] of [["litellm-yaml", yaml], ["litellm-py", py]]) {
    $(id).textContent = text_;
    document.querySelector(`[data-copy-target=${id}]`).dataset.copy = text_;
  }

  facts($("facts-self"), [
    ["Upstream this page dials", s.upstream || null],
    ["Mode", s.mode === "proxy" ? "in the request path (recording)" : "watching /health, /v1/models, /metrics only"],
    ["Poll", isnum(s.poll_s) ? `${s.poll_s} s` : null],
    ["Records kept", String(s.requests_kept ?? 0)],
    ["Series read", isnum(s.metrics_seen) ? `${s.metrics_seen} vllm:* series` : null],
  ]);
  text("updated", `Read ${clock(s.at)}`);
}

async function load() {
  try {
    const [s, host] = await Promise.all([api("/api/status"), api("/api/host")]);
    status = {...s, host};
    $("api-key").value = apiKey;
    render();
    notice("");
  } catch (e) {
    notice(e.message);
    text("updated", "Disconnected");
  }
}

function notice(message, info) {
  const el = $("error");
  el.hidden = !message;
  el.className = `notice${info ? " notice--info" : ""}`;
  if (message) el.textContent = message;
}

$("key-btn").onclick = () => { $("api-key").focus(); $("api-key").select(); };
$("key-save").onclick = async () => { setKey($("api-key").value); await load(); };
$("api-key").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); $("key-save").click(); } };
document.addEventListener("need-key", () => $("api-key").focus());
bindTheme();

(async function loop() {
  try { await load(); } catch (e) { notice(e.message); }   // load() catches; this keeps the finally running
  finally { setTimeout(loop, 5000); }
})();
document.addEventListener("visibilitychange", () => { if (!document.hidden) load().catch(() => {}); });
