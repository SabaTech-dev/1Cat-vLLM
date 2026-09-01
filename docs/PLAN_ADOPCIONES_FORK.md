# Plan de adopciones para el fork SM70 de 1Cat-vLLM

> Fecha: 2026-09-01 · Estado: PROPUESTO · Autor: análisis de repos externos (nano-vllm, vLLM upstream, vllm-omni, recipes.vllm.ai)
>
> Contexto: vLLM upstream abandonó Volta (docs exigen CC ≥ 7.5; consenso de comunidad: v0.19 fue la
> última línea práctica para V100; catch-22 documentado en vllm-project/vllm#43561). Nuestro fork
> (SabaTech-dev/1Cat-vLLM) cubre un nicho real: V100-SXM2 2×32GB, TP2, modelos Qwen3, contexto largo.

## Resumen del análisis de fuentes

| Fuente | Veredicto |
|---|---|
| [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (15.3k★) | Mina de oro para SM70: forks con atención sin FA2 (MinivLLM 1k★: paged attention Triton autocontenida; flex-nano-vllm 358★: FlexAttention funciona en SM70); prefix caching xxhash encadenado (~100 LOC) |
| vLLM upstream | Adopciones cherry-pick: Triton INT4 KV-quant (#40835, MERGED), EAGLE3-Qwen3 (#43132, MERGED), async scheduling (V1), KV watermark (#44594), LIFO free-block reuse (#51482), fused-MoE BLOCK_SIZE_K=64 en Volta (fix verificado) |
| [vllm-omni](https://github.com/vllm-project/vllm-omni) (6.5k★) | Sin adopción directa (multimodal, sin SM70). Lecciones: pinning de releases a una línea menor upstream; modelo de publicación tipo fork DGX-Spark (source build + patch set + receta de bench medido) |
| [recipes.vllm.ai](https://recipes.vllm.ai/) (181 recetas) | Cero contenido V100. Útiles: levers de memoria (block-size 256, gpu-mem-util 0.97–0.98, max-num-batched-tokens 2048–4096, max-num-seqs ~135–256), config EAGLE3, AWQ INT4 como única quant de pesos portable en Volta. No portable en SM70: checkpoints fp8, compute fp8/NVFP4, FA3. Nota: fp8_e5m2 KV **sí** funciona en este fork (validado en producción propia) |

## Restricciones y DoD transversal

- **DoD por sprint (no negociable)**: máximo 2 días de calendario; cierre solo con tests verdes +
  revisión de seguridad + benchmarks comparativos A/B + documentación curada.
- Ventana de GPU: los benches requieren ventana de mantenimiento (producción ocupa ambas V100 a 31 GB).
- **Hardware confound conocido**: la GPU 1 está en trámite de RMA (236.795 single-bit ECC aggregate,
  6× Xid 79). Anotar el estado del hardware en todo reporte de bench; benches finales post-RMA si es posible.
- Base del fork: rama `contrib/restore-sm70-build` (build SM70 del árbol 1.5.0-RC) sobre ef68a0ea.

## Sprint F1 — Atención Triton autocontenida (cimiento)

**Objetivo**: eliminar la dependencia de backends de atención que Volta no soporta bien, con un
backend paged-attention Triton propio (port adaptado de MinivLLM/nano-vllm), seleccionable como
`TRITON_PAGED`.

1. Port del kernel paged attention Triton (prefill varlen + decode con block tables) al layout del fork.
2. Registro del backend en el selector por-KV-grupo (patrón vLLM #48012).
3. A/B de calidad: logprob/MD5 canónico vs FLASH_ATTN_V100 y TRITON_ATTN en Qwen3.8-27B.
4. Bench de rendimiento: t/s decode y TTFT ×1/×4; escribir configs ganadoras.
5. Doc: `docs/backend-triton-paged-sm70.md` (cuándo usar cada backend).

**Riesgo**: Triton en SM70 es la capa portable pero poco optimizada; aceptar hasta −10% vs FA-lite
si la estabilidad lo compensa.

## Sprint F2 — KV cache: INT4 + higiene de bloques

**Objetivo**: duplicar contexto utilizable y reducir preemption.

1. Port del INT4 per-token-head KV-quant Triton (vLLM #40835) junto al fp8_e5m2 existente.
2. KV watermark para reducir preemptions (vLLM #44594).
3. LIFO free-block reuse con prefix caching off (vLLM #51482).
4. Bench: pool de KV tokens a 262K ctx, concurrencia máxima, tasa de preemption.
5. Doc: matriz de dtypes KV soportados en SM70 (fp16 / fp8_e5m2 / int4) con números.

## Sprint F3 — Async scheduling

**Objetivo**: solapar el scheduler CPU con el paso GPU (patrón V1) — ataque directo a la burbuja
de latencia TP2, sin dependencia de SM.

1. Habilitar/probar async scheduling del upstream sobre nuestra base (o backport mínimo).
2. Bench de latencia: TTFT p50/p99 y ITL bajo concurrencia, ×1/×4.
3. Doc: flags de scheduler recomendados para TP2 SM70 (con levers de recipes).

## Sprint F4 — Speculative decoding: EAGLE3-Qwen3 + prefix caching barato

**Objetivo**: mayor palanca de decode en contexto largo.

1. Verificar disponibilidad de draft heads EAGLE3 para nuestros tamaños de Qwen3; medir
   acceptance rate real en V100 (si < 0.5, descartar y quedarse con n-gram/dinámico).
2. Config EAGLE3 por receta (num_speculative_tokens 2–3) y bench t/s vs costo de compute.
3. Prefix caching xxhash encadenado de nano-vllm (~100 LOC) si el fork no lo cubre ya.
4. Doc: guía de speculative decoding en V100 con datos de aceptancia.

## Sprint F5 — Publicación del fork (modelo DGX-Spark)

**Objetivo**: convertir el trabajo interno en un fork público útil para la comunidad V100.

1. README público medido: qué soporta, qué no, benches reproducibles (con hardware y driver exactos).
2. Receta reproducible: build desde fuente, flags de serve validados, triage de problemas conocidos
   (custom all-reduce, Xid, backends de atención).
3. Tag de release estable del fork; rama de mantenimiento por línea menor upstream (pinning).
4. Anuncio en vllm-project/vllm#47019 (comunidad V100 activa) sin datos propietarios.

## Orden y dependencias

```
F1 (atención) ──> F2 (KV) ──> F4 (spec dec)
        │
        └──> F3 (scheduler, independiente)
F5 (publicación) al final, con los números de F1–F4
```

F1 es prerrequisito de F2/F4 (los benches de calidad pasan por el backend nuevo). F3 es
independiente y puede intercalarse si hay ventana de GPU.

## Fuera de alcance

- Todo lo multimodal (vllm-omni), compute fp8/NVFP4, FA2/FA3, bfloat16 tensor cores.
- MTP de Qwen3.8-27B en TP2 vLLM: descartado con datos (−52% decode medido; ver CHANGELOG interno).
