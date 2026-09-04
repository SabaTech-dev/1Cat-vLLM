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

#### Blueprint F2.1 v2 (INT4 KV per-token-head, listo para implementar)

INVESTIGACION COMPLETADA (2026-09-03): el fork ya tiene el patron
completo en TRITON_ATTN (`int8_per_token_head`) — NO reinventar:

1. **Estructura (espejo de triton_attn int8-PTH)**: cache de datos
   int4-empaquetado con head-slot `D/2 + 4 uint8` (los 4 bytes del
   final = escala fp32 por (block, slot, head), extraida con
   `torch.as_strided` sobre el untyped storage — ver
   `_ensure_scale_caches` de triton_attn.py:556). El spec presupuesta
   la memoria de escalas en `AttentionSpec.page_size_bytes` cuando
   `kv_quant_mode.is_per_token_head` (kv_cache_interface.py:166).
2. **Modo nuevo**: `KVQuantMode.INT4_PER_TOKEN_HEAD` +
   `get_kv_quant_mode("int4_per_token_head")` + branch en la capa
   (`get_kv_cache_spec` de attention.py) construyendo el spec con
   head_size empaquetado. CacheDType += "int4_per_token_head".
3. **Store**: cuantizacion simetrica por (token,head),
   `scale = max|k| / 7.0`, q en [-7,7], 2 nibbles por byte; hecho con
   torch ops en el wrapper de `do_kv_cache_update` (el store kernel es
   agnostico al dtype: head_dim = D/2+4 y tenentes uint8 pasan sin
   cambios). Escalas escritas via las vistas as_strided.
4. **Kernels**: `KV_INT4: tl.constexpr` en decode y prefill; load
   bytes [.., D/2] -> unpack lo/hi -> `tl.interleave` -> fp32 ->
   `* scale` broadcast (misma estructura del branch int8 de
   triton_unified_attention.py:850). Offsets con head-slot D/2+4.
5. **Gate**: "int4_per_token_head" solo en TRITON_PAGED (D par);
   fail-closed en otros backends. Prefix caching no toca bytes.
6. **Validacion**: sonda standalone (contiguo + unbind, patron F1) vs
   referencia; luego bateria PPL/greedy con kv-cache-dtype int4 y
   medicion de capacidad (objetivo ~3.5-3.9x tokens vs fp16).
7. **Alternativa descartada**: consumir turboquant_4bit_nc (4-bit MSE
   + norm correction en kernels TurboMind .so propietarios) — exige
   reverse-engineering del layout de slots; int4 propio es mas simple
   y basta para el objetivo de capacidad. int8_per_token_head (2x)
   ya disponible hoy para TRITON_ATTN si se quiere compression sin
   kernels nuevos.

#### RESULTADO F2.1 (2026-09-03): implementado, gate de calidad FALLA

La implementacion completa esta en `93f591250` (store cuantizado con
torch ops, store kernel con arange almohillado para slots de 68 bytes,
decode/prefill con unpack + descale, escalas in-slot, spec/allocation
uint8 sin tocar el alocador). La sonda standalone PASA (prefill 3e-4,
decode 2.5e-2 vs dequant). PERO el gate E2E en Qwen3-0.6B FALLA:
salida garbage con --kv-cache-dtype int4_per_token_head.

**Causa raiz (analisis)**: K post-RoPE tiene dimensiones outlier
(amax >> typical) — la escala simetrica por (token,head) amax/7 deja
SNR ~1 en las dimensiones no-outlier => la atencion se degrada a
ruido. La sonda sintetica (randn sin outliers) no puede detectarlo.
Es LA limitacion conocida del int4 simetrico por token-head.

**Caminos para promover (ordenado por esfuerzo)**:
a) Hibrido int8-K + int4-V: K sin outliers severos post-RoPE en int8
   (2x), V con distribucion mas pareja en int4 (~3.5x) => ~2.7x
   capacidad combinada; solo cambia el packing del store.
b) Escalas por-canal estaticas (calibradas) para las dims outlier +
   escala dinamica por token-head: mas storage de escalas.
c) Integrar TurboQuant 4-bit (kernels .so de TurboMind ya validados
   en FA-V100) — requiere consumir su formato desde Triton.

Mientras tanto el dtype queda EXPERIMENTAL (default fp16 intacto).

#### GATE E2E DEFINITIVO sobre el modelo de produccion (2026-09-03)

QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 (el checkpoint recomendado de la
release 1.5.0, 20 GB descargados) CARGA Y SIRVE en TRITON_PAGED TP1
V100 con NVFP4 weights + kernels SM70 del wheel 1.5.0. Resultados del
gate KV (dedicada, util 0.92, E5M2 no; fp16 vs int8-PTH):

| Metrica | fp16 KV | int8-PTH KV |
|---|---|---|
| PPL (corpus) | 2.5404 | 2.7344 (+7.6%) |
| Greedy coherente | 10/10 | 10/10 (0/10 byte-identico — ruido de cuantizacion) |
| **Tokens de KV cache** | **42,780** | **63,488 (+48%)** |
| Concurrencia max @4K | 10.44x | **15.50x** |

VEREDICTO: int8-PTH es FUNCIONAL en el modelo de produccion — salidas
coherentes, +48% de capacidad de contexto, coste PPL +7.6%. El +48%
(menor que el 1.94x teorico del shrink de K/V) viene del estado GDN
por-token no comprimible que acompana al KV en los hibridos Qwen3.8.
El fallo del 1.7B (PPL +14%, salida rota) era sensibilidad de escala:
los modelos menores degradan mas con KV int8. Queda experimental
(opt-in) a la espera de calibracion de escalas (per-channel estaticas)
para cerrar el gap de PPL.

#### Wheel 1.5.0 adoptado (2026-09-03)

El venv de desarrollo (/srv/benchmarks/1cat/venv) migra del wheel 1.3.0
al 1.5.0 oficial (`1cat_vllm-1.5.0-cp312`): resuelve el hibrido de
imports, alinea los kernels compilados (FlashAttention-V100, TurboMind
sampler, FlashQLA GDN sm70) y trae las mejoras XQA de la release. Los
.so del arbol del fork se refrescaron desde el wheel. Verificacion:
greedy 20/20 identico, PPL 6.5758, b8 +8.5% (2192 tok/s), b16 +4.3%,
prefill igual. Respaldo del estado 1.3.0:
`/tmp/opencode/venv-vllm-130-backup.tar.gz` (78 MB). El bug #457
(loop) es especifico del checkpoint QUASAR NVFP4 — no afecta nuestro
uso (modelos fp16 de desarrollo; el Qwen3.8-27B de produccion vive en
llama.cpp).

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

## Estado consolidado y riesgo de perdida (auditoria 2026-09-03)

### Inventario de piezas propias (rama sprint/f1-triton-attn sobre main)

| Pieza | Estado | Commits |
|---|---|---|
| TRITON_PAGED backend (kernels + registry) | VALIDADO (paridad + perf) | 242ed13f7, c91f09f71 |
| Tuning tiles (decode BN=128, prefill BM=16) | VALIDADO | a1ba06381 |
| CUDA Graphs decode-only | VALIDADO (3.4-4.2x decode) | 0c21c34b2 |
| LIFO free-block reuse | VALIDADO unit | 1f1c3828d |
| Watermark (port #44594) | VALIDADO smoke | db99c79e3 |
| int8-PTH KV | EXPERIMENTAL funcional (+48% ctx en QUASAR 27B; PPL +7.6%) | dee9d7060 |
| int8k/int4v + int4 simetrico | EXPERIMENTAL fail-closed (outliers RoPE) | 17db6425e |
| get_kv_cache_shape pad PTH | VALIDADO | dee9d7060 |
| QSA pre-Ampere | SUPERSEDED por #469 (eliminado) | — |

### Riesgos de perdida detectados y accion

1. **Tooling de validacion en /tmp (VOLATIL)** — RESUELTO: bateria,
   corpus, sondas y gate QUASAR versionados en
   `tools/sm70_validation/` con sus lecciones operativas.
2. **Hueco probe/E2E sin aislar** (int8-PTH: sonda pasa, E2E PPL
   +0.66 en 1.7B pero solo +7.6% aceptable en QUASAR 27B) — el
   aislamiento (prefix-reuse + multi-chunk + slots dispersos en la
   sonda) queda como primer item de F2.4.
3. **Syncs futuros**: upstream main avanza rapido (PLE, E4M3 QSA KV,
   MTP5). Cada sync debe preservar los 15 commits propios — rebase con
   conflicto esperado solo en arg_utils.py (flags propias).
4. **stash@{0}** obsoleto (stubs perdidos + CMakeLists; el build SM70
   lo cerro upstream #420) — dropear sin remordimiento.
5. **Resultados de gates solo en JSON de /tmp** — los numeros clave
   estan en este documento y en Engram; los JSON crudos se regeneran
   con el tooling versionado.

### Conflictos y sinergias entre sprints (nuevo)

- **F4 (spec decode) vs hallazgo humanjesse**: spec-decode es
  net-negativo en V100 hoy (flash_attn_v100 sin graphs bajo spec ->
  46 tok/s; triton_attn 77 vs 100 sin spec). F4 debe RE-ENFOCARSE:
  la via prometedora es comprobar si TRITON_PAGED sostiene graphs
  decode-only bajo verify multi-token (nuestro path prefill cubre
  multi-token) antes de invertir en EAGLE3.
- **F8 toolchain vs triton 3.5.1**: humanjesse documenta decode MLA
  ~3x mas lento con triton 3.6.0 en V100. Benchmarkear TRITON_PAGED
  bajo 3.5.1; si gana, pinear.
- **F9 vs upstream #464-473** (AWQ Qwen3.8 SM70, en revision): si
  aterrizan, parte del port F9 (kernels W4A16) puede venir de
  upstream; el hibrido y GGUF seguirian siendo nuestros.
- **F2.4 (bench 262K)**: usar la metodologia de #474 (concurrencia
  no-MTP TP4 con kernels por evidencia) adaptada a TP1.

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

1. **QSA dispatch pre-Ampere — ADOPTADO, CORREGIDO (2026-09-03)**:
   ALCANCE: QSA solo existe en la familia **Qwen3.8-Flash-Next**
   (qwen4_exp, MoE A4B de 48 capas 36 GDN + 12 QSA) — el 27B denso
   (QUASAR) NO tiene capas QSA y no se beneficia de este tuning. Las
   mediciones de #441 son sobre Flash-Next-180B en producción.
   CORRECCIÓN de ruta: el archivo vivo en CUDA es
   `qwen4_exp/nvidia/ops/qsa.py` (el árbol `amd/` es la vía ROCm — la
   nota original de #441 estaba invertida; el primer parche tocó el
   archivo equivocado y fue re-ubicado tras el rebase, commit
   0373df2ce). Ahora: rama `is_sm70` en la tabla de dispatch del
   archivo vivo con tiles estrechos de 4 warps (16/8/4, 16/4/4, 16/1/4
   según base_programs). Datos finales de Peuqui: narrow-16 gana TODOS
   los regímenes de prefill en V100 (1.2-2.6×) y RTX 8000 (1.16-1.20×)
   con numérica idéntica; el tile de 64 columnas ni siquiera lanza en
   SM75 a D=256 (OutOfResources vs 64 KiB smem). **SUPERSEDED**: su PR
   upstream #469 aterrizó y fue mergeado (2026-09-03 11:15) — el fork
   se sincronizó con origin/main (45a58ab67), nuestro parche local se
   eliminó y el dispatch #469 quedó validado 8/8 con los casos de test
   del propio PR.
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

### Sprint F9 — Modelos de producción en vLLM vía humanjesse/vllm-v100

DESCUBRIMIENTO (2026-09-03): humanjesse/vllm-v100 (fork de 1Cat ~0.0.3,
sin ancestro común con main actual, 23★) tiene AWQ-INT4 W4A16 FUNCIONAL
en V100 con flota verificada: Qwen3.6-27B-AWQ-INT4 (híbrido GDN — misma
familia arch que nuestro HauhauCS AWQ), MiniMax-M2.7-240B MoE, granite
8B, DeepSeek-V4-Flash W4A16, y GGUF servidos nativamente (Qwen3.6-35B
Q8 en 2×V100 ~100 tok/s con fixes RMSNorm/A_log/permute, Qwen3.5-122B
Q6_K en 8×V100, MiMo-310B, Mistral4-119B).

Piezas a portar (por tamaño):
1. `turbomind_asym.py` (213 líneas, archivo nuevo) —
   TurboMindAsymLinearKernel para W4A16 asimétrico vía `awq_gemm_sm70`
   (<0.1% error vs 2% del kernel Triton GPTQ que compone a garbage).
   Desbloquea la carga de HauhauCS AWQ-MTP (nuestro checkpoint de
   producción, W4A16 asimétrico).
2. Wiring compressed-tensors→AWQ SM70: delegación de
   CompressedTensorsSM70WNA16MoEMethod a AWQSM70MoEMethod + manejo de
   ignore-list y tuple-shard en el loader de qwen3_5.
3. GGUF classes qwen3_5_moe con los fixes críticos: RMSNorm Gemma-style
   `+1` (llama.cpp lo hornea en los pesos), `A_log` ya-exponentiado,
   permute de value-heads GDN, replicación KV para TP > nkv.
4. Regression runner de su flota verificado como harness.

#### F9.1 RESULTADO (2026-09-03): HauhauCS AWQ-MTP SIRVE en vLLM/V100

El checkpoint de produccion (twolven Qwen3.8-27B-Uncensored-HauhauCS-
Aggressive-AWQ-MTP, 20 GB) CARGA Y SIRVE en TRITON_PAGED con
`VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1` (el scheme WNA16 de 1.5.0
baja su min capability a 70 con el env; el gate "Min capability 75"
documentado antes era solo el default del env, no un limite de
hardware). OOM bajo captura de graphs a util 0.92/4096 -> sirve en
eager con util 0.97/2048 (tuning de memoria pendiente).

Gates de calidad (bateria, TP1 eager):
- AWQ + KV fp16: PPL 3.0414, greedy coherente y de alta calidad
  ("una de las ciudades mas visitadas del mundo. Sus ...") -> SALUDABLE
- AWQ + KV int8-PTH: PPL 215, 0/20 -> ROTO para este checkpoint
  (funcionaba en QUASAR NVFP4 +7.6%): la degradacion int8-PTH es
  DEPENDIENTE DEL MODELO, no un bug universal del kernel. Sospechoso:
  distribucion K/V del finetune HauhauCS (aggressive) bajo escala
  simetrica amax/127.

Configuracion de produccion recomendada para este checkpoint:
TURBOMIND=1 + KV fp16. El int8-PTH requiere calibracion (per-channel)
antes de usarse en modelos sensibles.

#### F2.4 (2026-09-03): calibracion int8 entregada + matriz de dtypes F2

Calibracion de escalas: el fallo int8-PTH en HauhauCS NO era NaN/Inf
(diagnostico: 0 filas afectadas) sino outliers intra-fila: amax p50=
0.0, p99=8.79, max=89.7 por (token,head) -> la escala amax/127 deja
~1% de resolucion a los canales de la masa. Fix: clip de percentil
(k-th mayor con suelo 5% de amax), env VLLM_SM70_INT8_CLIP, default
1% (commit 7bbb3ac94). Resultados (PPL, fp16 como referencia):
- HauhauCS 27B AWQ: 215.2 (amax) -> 4.21 (clip 1%), fp16 3.04 (+38%)
  [sweep: 0.2%->4.36, 1%->4.21, 5%->4.46; optimo ~1%]
- QUASAR 27B NVFP4: 2.73 (amax) -> 2.71 (clip 1%), fp16 2.54 (+6.7%,
  mejora vs +7.6% del amax). El clip es ganancia estricta en ambos.

Matriz de dtypes KV (F2 cerrada):
| dtype | QUASAR NVFP4 | HAUHAUCS AWQ |
| fp16 | PPL 2.54, ctx 42780 | PPL 3.04 |
| int8+clip1% | PPL 2.71 (+6.7%), ctx 63488 (+48%) | PPL 4.21 (+38%) |
| int4 sim | fail (outliers post-RoPE) | fail |
| int8k_int4v | fail-closed (slot uniforme) | fail-closed |

Graphs + AWQ TurboMind: la memoria cabe (batched-tokens 1024,
max-seqs 16, util 0.92) PERO la salida se corrompe (PPL 1.6M vs eager
3.04): el GEMM TurboMind usa workspace cacheado por stream
(StreamWorkspaceKey en el binario) y no es graph-safe. AWQ = eager
obligatorio hasta que el kernel se haga graph-safe (candidato a report
upstream). Graphs solo para el camino NVFP4.

Hallazgo de escalado (pendiente dedicar sesion): el prefill varlen de
TRITON_PAGED no completa en tiempo razonable a partir de ~16K tokens
de contexto en TP1 eager (6.1K verificado OK; 16K/32K/55K > 20-25 min
sin progreso observable). La capacidad de KV (63K int8) queda asi
inaccesible en un solo contexto hasta diagnosticar el kernel/scheduler.
La metodologia #474 (262K) requiere este fix previo.

#### Diagnostico del stall (2026-09-03, sesion issues)

Patron confirmado: GPU 100% con progreso CERO -> kernel colgado, no
scheduler. Faulthandler (in-process, 32K ctx): el stack del engine
cayo dentro de la cadena de muestreo greedy fundida
(qwen3_5.get_top_tokens -> logits_processor:319 ->
lm_head.maybe_get_sm70_lm_head_top1 -> op sm70_f16_lm_head_top1_out
sin retornar). Con VLLM_SM70_LM_HEAD_TOP1=0 el stall persiste (GPU
100%, 0%), pero esa corrida quedo en modo subprocess y el stack del
engine no se capturo -> inconcluso. Siguiente sesion: py-spy al
EngineCore con sudo (el dump al padre solo muestra queue.get),
probar tambien TOP1_TC=0 y capturar el segundo dump. Sospechoso
principal: familia de ops fundidas del LM head a contexto largo.
Nota operacional: llama-second se auto-restaura (watchdog) y compite
por GPU 1 - verificar memory.used=0 ANTES de cada ventana; los logs
deben ir a /tmp/opencode (/srv/benchmarks no es escribible por joker).
Issues revisadas: #478 (DFlash2 acceptance=0 TP4, spec-decode roto en
SM70 - confirma re-scope de F4), #424 (MTP crash a contexto
extendido), #214 (fused layernorm false-positive MTP). #334 respondida
con nuestra receta validada (env TURBOMIND + eager + memory tuning +
clip).

#### Diagnostico del stall, sesion 2 (2026-09-04): sospechoso principal

py-spy al EngineCore (el proceso se llama VLLM::EngineCore - los greps
de python no lo encuentran). Tres estados capturados en corridas 32K:
1. Forward activo moviendose por capas GDN (qwen3_next rms_norm) -
   el prefill AVANZA durante ~10+ min (lento, no colgado).
2. Cuelgue REAL confirmado dos veces: thread clavado en
   sm70_trace_event_sync (sm70_decode_trace.py:83) ->
   event.synchronize() del async_copy_ready_event que NUNCA dispara =
   un kernel del forward no termino (GPU 100% sin progreso).
3. Triton JIT compilando (_init_handles) - minutos extra por corrida.

Exonerado: el camino de muestreo top1 fundido (el stack previo era del
proceso PADRE, enganoso). Sospechoso principal: kernel FlashQLA-SM70
TileLang del GDN prefill (qwen_gdn_linear_attn.py:373 "original
FlashQLA-SM70 TileLang GDN prefill path") a contexto largo.

Pendiente definitivo: forzar --gdn-prefill-backend triton (el kwarg
gdn_prefill_backend y additional_config via LLM() NO llegaron al
resolver - "requested=auto" en ambos intentos; probar EngineArgs
directo o CLI serve). Si el backend triton completa 32K/55K, el fix
para la metodologia 262K es el cambio de backend GDN + reporte
upstream del kernel FlashQLA TileLang.

Leccion operacional critica: pkill -9 -f <patron> MATA EL SHELL PROPIO
si el patron aparece en su cmdline - explica fallos silenciosos
previos. Usar [c]orchetes o PID explicito.

#### ROOT CAUSE CONFIRMADO (2026-09-04, issue #488)

El kernel FlashQLA-SM70 (TileLang) del GDN prefill SE CUELGA a partir
de ~16K tokens de prompt; el backend Triton/FLA alterna COMPLETA:
32,760 tokens de prefill + 32 decode en 18m56s con salida coherente y
correcta via vllm serve --gdn-prefill-backend triton. Issue upstream
publicada (#488) con repro, stacks py-spy y workaround. El override
via CLI funciona (el kwarg LLM() se descarta en silencio; para offline
hay que usar el server o EngineArgs -> create_engine_config).

Consecuencia para la metodologia 262K: DESBLOQUEADA tecnicamente
(backend triton, prefill ~29 tok/s en TP1 - lento pero correcto);
el fix upstream del kernel FlashQLA restauraria la velocidad.

Notas operacionales de la ventana: los procesos hijos de un tool-call
mueren con el grupo al agotar el timeout (usar setsid nohup); la CLI
vllm serve del venv usa el wheel (necesita PYTHONPATH al repo para
nuestro backend TRITON_PAGED). Produccion :8009/:8001 OK.

Hallazgos transferibles documentados en #441/humanjesse:
- Spec-decode en V100 es net-negativo hoy (flash_attn_v100 no sostiene
  graphs bajo spec → PIECEWISE ~46 tok/s; triton_attn ~77 vs ~100
  sin spec) — impacta el diseño de F4.
- triton 3.6.0 genera código de decode MLA ~3× más lento en V100 que
  3.5.1 (ellos pinean 3.5.1) — benchmarkear nuestro TRITON_PAGED bajo
  3.5.1 antes de descartar.
- flash_attn_v100 dense prefill tenía un bug de aliasing sP/sS que
  ellos fixearon (be0fe44b) — verificar si nuestro vendor lo hereda.

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
