# Estado del fork — SabaTech-dev/1Cat-vLLM

> Última actualización: 2026-09-18 (campaña sabatech-sm70-qflash)

## Nuestra posición de serving

- **Prod sirve con el tag `v1.5.0`** (sha d8f42b3, 2026-09-02) = último release del upstream
  1CatAI/1Cat-vLLM. Verificado en el venv de prod (/srv/benchmarks/1cat, dist-info 1.5.0).
- Este fork org quedó **stale en 2026-08-31** (commits pre-release: "Record final SM70 wheel
  API gate", "Restore FastAPI metrics compatibility") — SIN sincronizar con v1.5.0 ni con main.

## Estado del upstream (a la fecha)

- main lleva ~2 semanas de fixes SM70 post-v1.5.0 sin cortar v1.6.0. El relevante para nosotros:
  **#645 "Decouple the Mamba state grid from the KV block size"** — requisito para servir
  híbridos GDN+MoE (qwen35moe, clase Qwen3.6-35B-A3B) vía vLLM.
- `fused_moe` SM70 YA está en v1.5.0 (47 referencias de código) → **MoE puro soportado hoy**.
  El hueco es solo el híbrido GDN.

## Estrategia acordada

1. Prod denso sigue en v1.5.0 (estable, sin acción).
2. Trackear el changelog de **v1.6.0**: si trae #645, evaluar upgrade para serving de híbridos.
3. Si servimos qwen35moe vía vLLM antes de v1.6.0 → build desde upstream main (o sincronizar
   este fork) ese día, con las envs SM70 del rig (`tools/eval_nvfp4_prod.sh` de qflash).
4. Sincronizar este fork con upstream al hacer cualquiera de las dos cosas anteriores.

## Números de referencia nuestros (2026-09-18, protocolo tg256/pp de Redhatvale)

- Prod denso (lyf-vl fp8-q4km, 1.5.0): tg256 **59.6 avg / 60.2 peak** · prefill ~1050 t/s @572pt
- NVFP4 QUASAR+dflash (single-stream): **124 tok/s** (2x el mejor stack documentado por terceros)
- Multi-stream: 6 seqs x 262144 ctx verificado (W9), continuous batching ON.

## Notas de la caja

- systemd: el unit `llama-vllm-star` cambió en disco — `daemon-reload` pendiente.
- Duelo de toolkits: el motor qflash compila y pasa verbatim con CUDA 12.9.2 y 12.8.2.
