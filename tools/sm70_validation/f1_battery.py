"""Bateria F1: quality gates + throughput para un backend de atencion.
GPU dedicada (solo este modelo cargado)."""

import argparse, json, math, time, sys

sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
from vllm import LLM, SamplingParams, TokensPrompt

PROMPTS = [
    "La capital de Francia es",
    "The capital of France is",
    "La fotosintesis es el proceso por el cual",
    "Un transformador electrico transfiere energia",
    "The theory of general relativity states that",
    "En 1969, el ser humano llego a la Luna",
    "El sistema nervioso humano esta compuesto por",
    "Deep learning models are trained using",
    "La economia circular propone",
    "Water treatment plants purify water by",
    "El guion del cortometraje comienza cuando",
    "La receta de la tortilla de patatas requiere",
    "Quantum computing uses qubits to",
    "Los beneficios del ejercicio fisico incluyen",
    "The Mediterranean diet is based on",
    "El telescopio espacial observo por primera vez",
    "La plasticidad sinaptica constituye la base",
    "Central banks influence markets through",
    "El cambio climatico provoca",
    "Reusable rockets reduce the cost of",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--util", default="0.85")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--kv-cache-dtype", default=None)
    args = ap.parse_args()

    res = {"backend": args.backend, "model": args.model}

    extra = {}
    if args.kv_cache_dtype:
        extra["kv_cache_dtype"] = args.kv_cache_dtype
    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=args.eager,
        attention_backend=args.backend,
        max_model_len=4096,
        gpu_memory_utilization=float(args.util),
        **extra,
    )
    tok = llm.get_tokenizer()

    # ---------- 1. greedy match ----------
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=64))
    res["greedy"] = [o.outputs[0].text for o in outs]

    # ---------- 2. logprobs de tokens generados ----------
    outs_lp = llm.generate(
        PROMPTS, SamplingParams(temperature=0, max_tokens=64, logprobs=1)
    )
    res["gen_logprobs"] = [
        [list(lp.values())[0].logprob for lp in o.outputs[0].logprobs] for o in outs_lp
    ]

    # ---------- 3. perplejidad sobre corpus fijo ----------
    corpus = open("/tmp/opencode/f1_corpus.txt").read()
    ids = tok.encode(corpus)
    CH = 512
    chunks = [ids[i : i + CH] for i in range(0, len(ids) - CH + 1, CH)][:12]
    outs_ppl = llm.generate(
        [TokensPrompt(prompt_token_ids=c) for c in chunks],
        SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0),
    )
    ppls = []
    for o in outs_ppl:
        lps = [list(lp.values())[0].logprob for lp in o.prompt_logprobs[1:]]
        ppls.append(math.exp(-sum(lps) / len(lps)))
    res["ppl_chunks"] = ppls
    res["ppl_mean"] = sum(ppls) / len(ppls)

    # ---------- 4. throughput ----------
    long_ids = (ids * 12)[:2048]
    llm.generate(
        [TokensPrompt(prompt_token_ids=long_ids[:512])],
        SamplingParams(max_tokens=1, temperature=0),
    )
    t0 = time.time()
    llm.generate(
        [TokensPrompt(prompt_token_ids=long_ids)],
        SamplingParams(max_tokens=1, temperature=0),
    )
    t1 = time.time()
    res["prefill_2048_ms"] = (t1 - t0) * 1000

    sp_b1 = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
    llm.generate(PROMPTS[:1], sp_b1)
    t0 = time.time()
    llm.generate(PROMPTS[:1], sp_b1)
    t1 = time.time()
    res["decode_b1_tps"] = 256 / (t1 - t0)

    sp_b8 = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
    llm.generate(PROMPTS[:8], sp_b8)
    t0 = time.time()
    llm.generate(PROMPTS[:8], sp_b8)
    t1 = time.time()
    res["decode_b8_tps"] = 8 * 256 / (t1 - t0)

    llm.generate(PROMPTS, sp_b8)
    t0 = time.time()
    llm.generate(PROMPTS, sp_b8)
    t1 = time.time()
    res["decode_b16_tps"] = 16 * 256 / (t1 - t0)

    json.dump(res, open(args.out, "w"))
    print("BATTERY DONE", args.backend, args.model)
    print("ppl_mean:", round(res["ppl_mean"], 4))
    print("prefill_2048_ms:", round(res["prefill_2048_ms"], 1))
    print("decode_b1_tps:", round(res["decode_b1_tps"], 1))
    print("decode_b8_tps:", round(res["decode_b8_tps"], 1))
    print("decode_b16_tps:", round(res["decode_b16_tps"], 1))


if __name__ == "__main__":
    main()
