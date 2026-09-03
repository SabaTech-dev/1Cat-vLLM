"""Gate E2E de F2.1 sobre QUASAR Qwen3.8-27B-NVFP4 (modelo de produccion)."""
import sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")


def main():
    import argparse, json, math

    ap = argparse.ArgumentParser()
    ap.add_argument("kv", choices=["fp16", "int8"])
    KV = ap.parse_args().kv

    from vllm import LLM, SamplingParams, TokensPrompt

    MODEL = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
    extra = {"kv_cache_dtype": "int8_per_token_head"} if KV == "int8" else {}

    llm = LLM(model=MODEL, dtype="float16", enforce_eager=True,
              attention_backend="TRITON_PAGED", max_model_len=4096,
              gpu_memory_utilization=0.92, trust_remote_code=True, **extra)
    tok = llm.get_tokenizer()

    PROMPTS = [
        "La capital de Francia es",
        "The capital of France is",
        "La fotosintesis es el proceso por el cual",
        "Un transformador electrico transfiere energia",
        "The theory of general relativity states that",
        "El sistema nervioso humano esta compuesto por",
        "Deep learning models are trained using",
        "La economia circular propone",
        "La receta de la tortilla de patatas requiere",
        "Quantum computing uses qubits to",
    ]

    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=48))
    greedy = [o.outputs[0].text for o in outs]
    for p, t in zip(PROMPTS[:3], greedy[:3]):
        print(f"[{KV}] {p[:32]} -> {t[:70]!r}")

    corpus = open("/tmp/opencode/f1_corpus.txt").read()
    ids = tok.encode(corpus)
    CH = 512
    chunks = [ids[i:i + CH] for i in range(0, len(ids) - CH + 1, CH)][:8]
    outs_ppl = llm.generate([TokensPrompt(prompt_token_ids=c) for c in chunks],
                            SamplingParams(max_tokens=1, temperature=0,
                                           prompt_logprobs=0))
    ppls = []
    for o in outs_ppl:
        lps = [list(lp.values())[0].logprob for lp in o.prompt_logprobs[1:]]
        ppls.append(math.exp(-sum(lps) / len(lps)))
    ppl_mean = sum(ppls) / len(ppls)
    print(f"PPL[{KV}]: {ppl_mean:.4f}")

    json.dump({"kv": KV, "greedy": greedy, "ppl_mean": ppl_mean,
               "ppl_chunks": ppls},
              open(f"/tmp/opencode/f2-quasar-{KV}.json", "w"))
    print(f"QUASAR GATE DONE {KV}")


if __name__ == "__main__":
    main()
