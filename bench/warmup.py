#!/usr/bin/env python3
"""Compile the size-class variants of the lazily-JIT'd Triton kernels (QSA sparse attention picks its tile config
from the token count: <=8, <32, <=256, <=512, >512; vLLM's own warmup covers only the extremes) right after
startup, so the first real prompt in each class does not pay ~2.4 s of compilation. Unique prompts (no prefix hits).
usage: warmup.py [BASE=http://localhost:8080]"""
import json, os, random, sys, time, urllib.request
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080") + "/v1/chat/completions"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from served_model import resolve          # the model name comes from the serve, never from this file
MODEL = resolve(BASE)
t0 = time.time()
for words in (12, 60, 180, 400, 1500):
    p = " ".join(f"w{random.randint(0, 10**6)}" for _ in range(words))
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 2,
            "chat_template_kwargs": {"enable_thinking": False}}
    urllib.request.urlopen(urllib.request.Request(BASE, json.dumps(body).encode(),
                                                  {"Content-Type": "application/json"}), timeout=600).read()
print(f"warmup done in {time.time()-t0:.1f}s")
