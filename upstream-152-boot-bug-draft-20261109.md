# DRAFT issue upstream — v1.5.2 wheel boot crash (NO postear sin GO del dueño)

> Estado: DRAFT para revisión de Joker. Si da GO, publicar en 1CatAI/1Cat-vLLM (ref: flujo #865).
> Evidencia completa: /tmp/opencode/ab152/legs/ (journal-B4-fail.log, journal-C-boot-fail.log) + dumps.

**Title**: `[Bug] 1.5.2 wheel fails to boot with quantized lm_head — AttributeError: 'ParallelLMHead' object has no attribute '_layer_execution_policy'`

**Environment**
- 2× V100-32GB (PCIe), TP=2, driver 580.178.04, Ubuntu, Python 3.12
- Wheel: `1cat_vllm-1.5.2+cu128-cp312-cp312-linux_x86_64.whl` (SHA256 a27afc51…, verified)
- Model: `coder390/Qwen3.8-27B` NVFP4 W4A16 (community quantization; **lm_head is quantized**)
- Command: `vllm serve <model> --tensor-parallel-size 2 --max-model-len 200000 --max-num-seqs 16 --kv-cache-dtype fp8_e5m2|fp8_e4m3 --gpu-memory-utilization 0.90 --max-num-batched-tokens 8192 --enable-prefix-caching --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`

**Symptom**: deterministic boot crash — 6/6 attempts over 30 min (engine reaches deep init: SM70 auto-settings, compile policies, "Enabled custom fusions: norm_quant"), then WorkerProc fails:

```
File ".../vllm/model_executor/layers/linear.py", line 375, in process_weights_after_loading
    policy=layer._layer_execution_policy,
File ".../vllm/model_executor/layers/vocab_parallel_embedding.py", line 340, in __getattr__
AttributeError: 'ParallelLMHead' object has no attribute '_layer_execution_policy'
```

- Same failure with `--kv-cache-dtype fp8_e5m2` and `fp8_e4m3` → **not KV-dtype related**.
- Root mechanism: `_layer_execution_policy` is assigned at `linear.py:489` (`self._layer_execution_policy = layer_policy()`), which `ParallelLMHead` (quantized LM head routed through the ModelOpt/TurboMind dense path) never executes.
- Log line right before the crash: `[modelopt.py:1134] SM70 ModelOpt NVFP4 TurboMind dense path enabled (weight-only; activations remain half).`
- Hypothesis: the reference checkpoint (Unsloth Qwen3.8-27B-NVFP4) keeps `lm_head` in half precision, so the release never exercises a quantized `ParallelLMHead` through this path. Checkpoints with quantized `lm_head` (community quants) crash at boot.
- Same checkpoint boots fine on our v1.5.1-era dev build (sprint branch), so it is a 1.5.2 regression in the weight-processing path for quantized LM heads.

**Suggested fix direction**: guard `linear.py:375/388/428` with `getattr(layer, "_layer_execution_policy", None)` (or assign the policy in `ParallelLMHead` construction when quantized), or route quantized LM heads through the legacy process-weights path.

**Full logs**: `journal-B4-fail.log` (6 attempts, e5m2), `journal-C-boot-fail.log` (e4m3) — available on request.
