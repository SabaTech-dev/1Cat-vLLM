"""F2.4: throughput y sanity a contexto largo (metodologia #474, TP1).

Prefill masivo (corpus repetido) + decode greedy: mide tok/s de decode
en el borde de capacidad de cada dtype de KV.
"""

import faulthandler
import sys
import time

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")

MODEL = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"


def main() -> None:
    faulthandler.dump_traceback_later(240, repeat=True, file=sys.stderr)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kv_dtype = sys.argv[1]  # float16 | int8_per_token_head
    target_ctx = int(sys.argv[2])  # tokens de prefill
    out_path = sys.argv[3]

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    corpus = open("/tmp/opencode/f1_corpus.txt").read()
    ids = tok.encode(corpus)
    reps = target_ctx // len(ids) + 1
    prompt_ids = (ids * reps)[:target_ctx]
    print(f"prompt: {len(prompt_ids)} tokens (kv={kv_dtype})", flush=True)

    llm = LLM(
        model=MODEL,
        dtype="float16",
        attention_backend="TRITON_PAGED",
        max_model_len=target_ctx + 4096,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
        enforce_eager=True,
        kv_cache_dtype=kv_dtype if kv_dtype != "float16" else "auto",
    )

    t0 = time.time()
    out = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        SamplingParams(temperature=0, max_tokens=32),
    )[0]
    dt = time.time() - t0
    text = out.outputs[0].text[:80].replace("\n", " ")
    print(f"total {dt:.1f}s | OUT: {text}", flush=True)

    import json

    json.dump(
        {"kv": kv_dtype, "ctx": len(prompt_ids), "total_s": round(dt, 1), "out": text},
        open(out_path, "w"),
    )
    print("F24 DONE", flush=True)


if __name__ == "__main__":
    main()
