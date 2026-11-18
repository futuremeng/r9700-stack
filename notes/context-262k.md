# `context-262k.md` was renamed → [context-256k.md](context-256k.md)

The old name rounded the token count in decimal and then attached a `k` that means ×1024 everywhere else in this
repo. **262,144 tokens is 256k.** So as of 2026-11-18 the note, this box's launch config and the launcher's `CTX=`
profile all say `256k`, and `CTX=262k` is still accepted, printing that it was renamed.

Nothing was re-measured and no value moved: every number in that note was always taken at `--max-model-len 262144`.
The stub exists because CHANGELOG, PROGRESS and several dated notes say "262k" — which is what they said at the time,
and rewriting a dated record to fix a label would make the record wrong instead of the label right.
