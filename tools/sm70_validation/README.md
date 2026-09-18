# Herramientas de validacion SM70 (sprints F1/F2)

Instrumental de las gates de calidad y benchmarks. Uso tipico en
VENTANA DEDICADA (solo el modelo de test en la GPU — regla del
operador) con el venv de desarrollo y `sys.path` al fork.

## Contenido

- `f1_battery.py` — bateria de quality gates: greedy (20 prompts),
  logprobs, PPL sobre `f1_corpus.txt` (TokensPrompt, sin re-encode) y
  throughput (prefill 2048, decode b1/b8/b16 con ignore_eos). Flags:
  `--backend --model --util --eager --kv-cache-dtype`.
- `f1_corpus.txt` — corpus fijo bilingue (~6k tokens) para PPL.
- `f1_autotune.py` — sweep de tiles decode/prefill con CUDA events
  (medianas de 20 corridas) en formas Qwen3-0.6B.
- `f1_kernel_probe.py`, `f2_int4_probe.py`, `f2_hybrid_probe.py` —
  sondas standalone de kernels vs referencia torch (patron: tensores
  contiguos + vistas unbind(1) no contiguas). La de hibrido inyecta
  outliers RoPE-style en K.
- `f2_quasar_gate.py` — gate sobre QUASAR Qwen3.8-27B-NVFP4 (modelo de
  la familia de produccion): greedy + PPL fp16 vs int8-PTH.

## Lecciones operativas (no saltarse)

1. Canarios de corrupcion: SIEMPRE acordes al API path del test (raw
   completion de un modelo BASE no es corrupcion; el gate es PARIDAD
   PPL/greedy vs fp16, no calidad absoluta del texto).
2. Scripts offline de vLLM requieren guard `if __name__ == "__main__"`.
3. Cuantizacion: las sondas sinteticas no detectan problemas de
   distribucion real (outliers post-RoPE) — el gate E2E con modelo
   real es irremplazable.
4. Salida de systemd-run va al journal de la unidad, no al redirect
   del cliente.
