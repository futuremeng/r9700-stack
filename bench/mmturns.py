#!/usr/bin/env python3
"""bench/mmturns.py - does the engine re-encode the pictures a chat client replays from history?

Two requests, and the pair is the measurement:

  turns   three messages: the first carries TWO images, the last one image. A client that replays its transcript
          (which is what every chat client does) sends all three pictures on the third turn. With stock behavior
          all three are processed; with compat/mm_last_turn.py only the one being asked about is, which shows up
          in usage.prompt_tokens as ~2 images' worth of tokens less, and in the serve's log line.
  five    one turn carrying FIVE images. `--limit-mm-per-prompt image:4` counts images per PROMPT, so stock answers
          400 "At most 4 image(s) may be provided in one prompt" -- a refusal of a request whose newest content is
          one picture. With the filter the newest four survive and the request answers 200.

Run it against a launch with R9K_MM_LAST_TURN=0 and again with it on: the difference is the patch's effect on the
serving line, not on a unit test's copy of the messages.

  python3 bench/mmturns.py                     # against http://localhost:8080
  BASE=http://other:8080 python3 bench/mmturns.py
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:8080")
IMG = os.environ.get("IMG", os.path.expanduser("~/r9700-mm-test.png"))
SERVED = os.environ.get("SERVED_NAME", "Qwen3.8")


def img_part():
    with open(IMG, "rb") as fh:
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(fh.read()).decode()}}


def post(payload):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return e.code, body[:220]


def main():
    txt = [{"type": "text", "text": "Describe this in three words."}]
    a = post({"model": SERVED, "max_tokens": 24, "temperature": 0,
              "messages": [{"role": "user", "content": txt + [img_part(), img_part()]},
                           {"role": "assistant", "content": "Noted."},
                           {"role": "user", "content": txt + [img_part()]}]})
    status, body = a
    pt = body.get("usage", {}).get("prompt_tokens") if status == 200 else None
    print(f"turns   http {status}  prompt_tokens {pt}"
          + ("" if status == 200 else f"  {body}"))
    b = post({"model": SERVED, "max_tokens": 24, "temperature": 0,
              "messages": [{"role": "user", "content": txt + [img_part() for _ in range(5)]}]})
    status2, body2 = b
    pt2 = body2.get("usage", {}).get("prompt_tokens") if status2 == 200 else None
    print(f"five    http {status2}  prompt_tokens {pt2}"
          + ("" if status2 == 200 else f"  {body2}"))
    if pt and pt2:
        print(f"read: the three-turn request carries 3 images and the single-turn one 5. Only the last turn's "
              f"image processed -> prompt_tokens of 'turns' ~ one image lower than stock; and 'five' answering "
              f"200 rather than 400 is the ceiling the replay used to hit by itself.")
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
