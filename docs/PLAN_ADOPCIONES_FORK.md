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

**ESTADO (2026-09-02): COMPLETADO Y PROMOVIDO.** Puntos 1-2 hechos (rama
`sprint/f1-triton-attn`); el 3-4 se ejecutó sobre Qwen3-0.6B fp16 (ver
limitación de modelo abajo) con GPU dedicada: PPL diff 1e-4 vs
FLASH_ATTN_V100, decode GANA (+16% b1, +15% b8, +7% b16), greedy 19/20,
tuning aplicado (decode BN=128, prefill BM=16; tiles grandes colapsan por
smem en Volta). Debilidad documentada: prefill largo ~3x detrás de FA.

**ACTUALIZACIÓN (2026-09-03)**: CUDA Graphs RESTABLECIDOS tras
re-evaluación (el revert del 09-02 fue un falso positivo de canario —
modelo base vs chat template, ver docs/f1-triton-paged-sm70.md):
greedy 20/20 idéntico a eager, PPL idéntica, decode 3.4-4.2× más rápido
(b1: 310 vs 73 tok/s; b16: 3309 vs 966). Soporte final:
`UNIFORM_SINGLE_TOKEN_DECODE`. El backend queda con paridad numérica
completa Y la mejor velocidad de decode medida en V100.

**LIMITACIÓN DE MODELO DESCUBIERTA**: el checkpoint HauhauCS Qwen3.8-27B
disponible en formato HF es compressed-tensors W4A16, cuyo scheme exige
capability >= 7.5 — NO carga en Volta (7.0); el BF16 local (~50 GB) no
cabe en 32 GB. Los A/B sobre la familia de producción exigen re-cuantizar
a GPTQ compatible SM70 o equivalentes; hasta entonces los gates se
validan con modelos fp16 que quepan.

**Riesgo**: Triton en SM70 es la capa portable pero poco optimizada; aceptar hasta −10% vs FA-lite
si la estabilidad lo compensa.

## Sprint F2 — KV cache: INT4 + higiene de bloques

**ESTADO (2026-09-03)**: puntos 2 y 3 HECHOS —
- Punto 2 (watermark): port de upstream vllm#44594 (mergeado jun-2026);
  `--watermark <fracción>` reserva bloques libres en admisiones de
  waiting/preempted (default 0.0 = off). Commit b1848928f. Smoke E2E OK.
- Punto 3 (LIFO): `--block-reuse-order lifo` (solo con prefix caching
  off; unit test de reuso). Commit 252d6b6b4.
Punto 1 (INT4 KV en kernels TRITON_PAGED) pendiente; punto 4 (bench 262K)
requiere ventana dedicada con carga decode-heavy cerca de capacidad.

**RETRACCIÓN (2026-09-03)**: el "bug de engine a util baja" de esta
misma noche era un FALSO POSITIVO — no existe. Qwen3-0.6B es un modelo
BASE y su completion raw de "La capital de Francia es" es "el
**Estados Unidos**..." SIEMPRE (transformers puro lo reproduce); el
"Paris" venia del path con chat template. Toda la matriz (util, backend,
graphs/eager, fifo/lifo) era consistente: nunca hubo corrupcion. Los
tests futuros deben usar canarios apropiados al API path o modelos
instruct. Por la misma razon se RESTAURARON los CUDA Graphs (ver
docs/f1-triton-paged-sm70.md): greedy 20/20 identico, decode 3.4-4.2x.

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

### Adopciones del issue #441 (usuario de producción, rig mixto Volta/Turing)

Peuqui reporta tuning medido en producción (2×RTX 8000 + 3×V100,
Qwen3.8 bajo llama-swap). Estado de adopción:

1. **QSA dispatch pre-Ampere — ADOPTADO (2026-09-03, d9646abcc)**: rama
   de capability en `qwen4_exp/amd/ops/qsa.py` (el archivo vivo en
   pre-Ampere; el gemelo `nvidia/` es código muerto ahí): tiles
   estrechos de 4 warps (16/8/4, 16/4/4, 16/1/4 según base_programs).
   Su medición: 19.3× en V100 para prefill de 2048 tokens (27.3 ms vs
   527 ms con los tiles GB300).
2. **flash_attn_v100 prefill tile 64×80 (−13 % en D=128) — PENDIENTE**:
   el tile 32×176 está horneado en el wheel compilado
   `flash_attn_v100`; hace falta la fuente del paquete (Peuqui ofrece
   diffs/PR). Restricción reportada: M múltiplo de 32 (48×112 y 80×48
   dan resultados erróneos).
3. **Verify multi-token**: el verify escala lineal con q (0.153/0.222/
   0.632 ms para q=1/2/8 en H=4/HK=1/D=128, 31k KV) — apalanca el spec
   decode en Volta; su fix de split-kv está propuesto upstream
   (flash-attention #190). Referencia para F4.
4. **TP heterogéneo sm70/sm75**: derivar la versión de FA por worker en
   vez de por device 0 (vllm #54758). Referencia para despliegues mixtos.

## Fuera de alcance

- Todo lo multimodal (vllm-omni), compute fp8/NVFP4, FA2/FA3, bfloat16 tensor cores.
- MTP de Qwen3.8-27B en TP2 vLLM: descartado con datos (−52% decode medido; ver CHANGELOG interno).

## Adopciones de segunda ola (deep research 2026-09-02)

Fuentes: FlashML-org/FreeToken (11.2k★), repos unslothai (fork llama.cpp,
unsloth-zoo, notebooks), investigación drivers/eco-sistema Volta.

### Sprint F6 — Prefix caching consciente de GDN + pools híbridos
- **Diseño FreeToken** (no ejecutable en Volta — CUDA 13 — pero es la
  referencia): `linear_state_pool` + `hybrid_radix_cache` — pools SEPARADOS
  para estado recurrente (GDN) y KV de atención completa con radix cache.
- El prefix caching del fork debe invalidar/reusar TAMBIÉN el estado
  lineal (no solo el KV paginado) en modelos híbridos Qwen3.8.
- **Semantic anchor checkpoints**: checkpoints de estado por fronteras
  semánticas (tool calls, bloques de razonamiento) — edits agénticos sin
  recomputar el prefill de 256K.

### Sprint F7 — Elastic memory manager
- Reasignación de VRAM en runtime entre caché de expertos y KV sin
  reiniciar (patrón FreeToken). En TP2 V100 32 GB + contextos 256K, la
  frontera KV/caché es la palanca de capacidad más grande.

### Sprint F8 — Iteración y toolchain (transversal)
- `compile_cache.py` de unsloth-zoo: caché persistente de torch.compile
  entre sesiones para acelerar el ciclo de desarrollo del fork.
- Toolchain fijada: CUDA 12.9 (último 12.x; 13.x eliminó sm_70) +
  `TORCH_CUDA_ARCH_LIST="7.0"` + driver R580 (LTSB hasta jun-2028, la
  rama final con Volta; NO migrar a 595/610).
- Triton: main exige CC 8.0+ — pinear 3.x con binarios sm_70 para el
  Sprint F1 (atención Triton autocontenida).

### Nota llama.cpp de producción (fuera de los dos repos, acción registrada)
- ~~Cherry-pick del pin #144 de unslothai/llama.cpp~~ OBSOLETO (2026-09-02):
  upstream mergeó qwen4exp nativamente (#27742) y master ya trae draft-mtp
  con auto-detección (#27005). Producción llama-second :8009 fue swap-eada
  directamente a master 866322481 (56.7 tok/s temp-0, +26% vs FastMTP;
  aceptación draft 0.90 vs 0.73). Detalle en Engram.
- Fix de una línea: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` (pin #149).
- WIP perdido: `csrc/local_sm70_stubs.cpp` era untracked y se perdió en el
  branch switch; solo sobrevive el hunk de CMakeLists en `stash@{0}`.
  Regenerar los stubs desde los link-errors del build SM70 en F2.
