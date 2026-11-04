#!/usr/bin/env python3
"""tests/test_prefix_replay.py -- the prefix-cache replay, tested against cases small enough to do by hand.

The tool's whole job is to answer "would a prefix cache pay for itself on the traffic this box carries", so the
tests are about the four things that could silently make it lie: a hit that is not a PREFIX, blocks that vanish
from the accounting, retained blocks that are counted as free, and a corpus that quietly loses part of itself.
"""
import importlib.util
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fail = 0


def ck(what, ok, extra=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + what + ("" if ok else "  " + extra))
    if not ok:
        fail += 1


spec = importlib.util.spec_from_file_location("pr", os.path.join(ROOT, "tools", "prefix-replay.py"))
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)

BL = 4                       # four-token blocks, so a hand-built case is a few characters long
VOCAB = {c: i for i, c in enumerate("abcdefghij")}


class Tok:
    """A tokenizer that maps each character to one token id: token counts are then character counts."""
    def encode(self, s):
        class E:
            ids = [VOCAB.get(c, 0) for c in s]
        return E()


tok = Tok()


def calls(*specs):
    """specs: (t, dur, session, prompt) -> the call shape load_sessions produces."""
    return [(t, d, s, tok.encode(p).ids, len(p), 0) for t, d, s, p in specs]


ck("blocks_of hashes whole blocks only: 8 tokens give two hashes, 9 tokens still give two (the tail is not shared)",
   len(pr.blocks_of(list(range(8)), 4)) == 2 and len(pr.blocks_of(list(range(9)), 4)) == 2
   and pr.blocks_of(list(range(3)), 4) == [], str(pr.blocks_of(list(range(9)), 4)))
# 1. one session, growing context: the second call's prefix is the first call's whole prompt
two = calls(("2026-01-01T00:00:00Z", 1.0, "s1", "abcdefgh"),
            ("2026-01-01T00:00:05Z", 1.0, "s1", "abcdefghijkl"))
on = pr.simulate(two, BL, 100, 1.0, True)
off = pr.simulate(two, BL, 100, 1.0, False)
ck("an agent that resends its own transcript gets its whole previous context back (8 of 12 tokens)",
   on["hit_tokens"] == 8 and on["requests"] == 2, str(on))
ck("with the cache off, nothing is shared at all (not even identical content)",
   off["hit_tokens"] == 0, str(off))

# 2. prefix, not substring: the shared content sits AFTER a different first block
sub = calls(("2026-01-01T00:00:00Z", 1.0, "s1", "abcdefgh"),
            ("2026-01-01T00:00:05Z", 1.0, "s1", "zzzzabcdefgh"))
ck("a shared block only counts as part of the CONTIGUOUS prefix (the engine stops at the first miss)",
   pr.simulate(sub, BL, 100, 1.0, True)["hit_tokens"] == 0,
   str(pr.simulate(sub, BL, 100, 1.0, True)))

# 3. a different session that shares nothing must evict what the first one left behind, and that eviction is
#    the visible half of the cost a hit rate hides: s1 finishes, keeps 2 blocks, and the pool is 2 wide.
diff = calls(("2026-01-01T00:00:00Z", 1.0, "s1", "abcdefgh"),
             ("2026-01-01T00:00:03Z", 1.0, "s2", "ijkl"))
tight = pr.simulate(diff, BL, 2, 1.0, True)
ck("retained blocks occupy the pool: a 2-block pool holding 2 retained blocks evicts before it can admit",
   tight["evictions"] >= 1 and tight["peak_blocks"] <= 2 and tight["hit_tokens"] == 0, str(tight))
ck("the same pair with the cache off never evicts (nothing was retained to evict)",
   pr.simulate(diff, BL, 2, 1.0, False)["evictions"] == 0)

# 4. retention cost is counted as occupancy, and OFF does not retain anything
hold = calls(*[(f"2026-01-01T00:00:{i:02d}Z", 1.0, "s1", "abcdefgh") for i in range(1, 8)])
r_on = pr.simulate(hold, BL, 4, 1.0, True)
r_off = pr.simulate(hold, BL, 4, 1.0, False)
# The last call in the fixture is still running when the replay ends, so "retained at end" is 0 in both runs --
# what the case proves is the reuse itself: 48 of 56 prompt tokens come back, and the prefill falls by that much.
ck("seven repeats of one transcript: ON computes 8 tokens, OFF computes all 56, and neither overflows the pool",
   r_on["peak_blocks"] <= 4 and r_off["peak_blocks"] <= 4 and r_on["hit_tokens"] == 48
   and r_off["hit_tokens"] == 0 and r_on["prefill_s"] * 6 < r_off["prefill_s"], f"{r_on} {r_off}")

# 5. the queue: when nothing is evictable, a request waits -- and the wait is reported, not hidden
q = calls(("2026-01-01T00:00:00Z", 60.0, "s1", "abcdefgh"),
          ("2026-01-01T00:00:01Z", 1.0, "s2", "ijklmnop"))
qr = pr.simulate(q, BL, 2, 1.0, False)
ck("a request that cannot be admitted waits for a running one to finish, and that wait is counted in seconds",
   qr["admission_waits_s"] > 50 and qr["requests"] == 2, str(qr))

# 6. the corpus reader survives the corrupt records this box's session files really contain
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
    fh.write(json.dumps({"type": "message", "timestamp": "2026-01-01T00:00:00Z",
                         "message": {"role": "system", "content": "",
                                     "sections": {"preamble": "ab", "tools": "cd", "rules": "", "docs": "", "cwd": ""}}}) + "\n")
    fh.write("\x00" * 40 + "\n")        # zero-filled record: what a write that died mid-flight leaves behind
    fh.write(json.dumps({"type": "message", "timestamp": "2026-01-01T00:00:05Z",
                         "message": {"role": "assistant", "content": [{"type": "text", "text": "ef"}],
                                     "durationMs": 900,
                                     "usage": {"input": 2, "output": 2, "cacheRead": 2}}}) + "\n")
    name = fh.name
cl, skipped = pr.load_sessions(name, tok)
os.unlink(name)
ck("a corrupt record is skipped and COUNTED, not taken as the end of the file",
   len(cl) == 1 and skipped == 1, f"{len(cl)} calls, {skipped} skipped")
ck("the prompt of the first call is exactly the system sections, joined in the harness's order",
   cl[0][3] == tok.encode("ab\ncd\n\n\n").ids, str(cl[0][3]))
ck("durationMs becomes seconds (the block-holding time the replay uses)",
   abs(cl[0][1] - 0.9) < 1e-9, str(cl[0][1]))

# 7. fidelity: the client's prompt count is input + cacheRead, not input alone
fid = pr.fidelity([(c[0], c[1], c[2], tok.encode("abcd").ids, 3, 5) for c in cl])
ck("fidelity compares against input + cacheRead (the upstream reports cached tokens separately)",
   fid and fid["median_ratio"] == 0.5, str(fid))

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
