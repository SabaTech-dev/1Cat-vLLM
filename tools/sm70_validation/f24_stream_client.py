"""Cliente streaming para F2.4: prefill 32K + 128 decode con timestamps."""

import json
import sys
import time

import requests

URL = "http://127.0.0.1:8321/v1/completions"
req = json.load(open("/tmp/opencode/req30k_stream.json"))
t0 = time.time()
first_token_ts = None
chunks = []

with requests.post(URL, json=req, stream=True, timeout=2400) as r:
    for line in r.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8", errors="replace")
        if line.startswith("data: ") and line != "data: [DONE]":
            now = time.time()
            if first_token_ts is None:
                first_token_ts = now
            chunks.append(now)

with open("/tmp/opencode/f24-stream-result.json", "w") as f:
    json.dump(
        {
            "t0": t0,
            "first_token_s": round(first_token_ts - t0, 1) if first_token_ts else None,
            "n_chunks": len(chunks),
            "decode_s": round(chunks[-1] - first_token_ts, 1)
            if len(chunks) > 1
            else None,
            "decode_tps": round((len(chunks) - 1) / (chunks[-1] - first_token_ts), 2)
            if len(chunks) > 2
            else None,
            "total_s": round(time.time() - t0, 1),
        },
        f,
    )
print("STREAM DONE", flush=True)
