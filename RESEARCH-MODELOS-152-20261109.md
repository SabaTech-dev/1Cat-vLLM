# RESEARCH: Modelos candidatos para 1Cat-vLLM 1.5.2 en jokerserver (2×V100-32GB PCIe, TP2)

> Fecha: 09-nov-2026 · Fuentes: RELEASE_NOTES v1.5.2 + PR_INDEX.md + README 1Cat-LLM (main) +
> HF orgs QUASAR-QAT / RadixArk / incoai (09-nov) + flags del wheel venv152.
> Restricción clave: la wheel v1.5.2 NO arranca checkpoints NVFP4 con **lm_head cuantizado**
> (bug `_layer_execution_policy`, 6/6 con coder390). Los candidatos de abajo lo evitan.

## Contexto del rig

- 2× V100-32GB PCIe (TP2, 64GB VRAM total), cap 200W, host RAM amplio, CUDA 12.9/nvcc ✓.
- La referencia del release usa TP4 (4×V100); nuestros números serán menores — el A/B es interno.
- Especulación disponible: DFlash2 (drafts en incoai/RadixArk) y MTP4 (checkpoints nativos).
- Presupuesto de descargas: /srv 260G libres ✓.

## T1 — Desbloquea la wheel para prod (prioridad máxima)

| Modelo | Tamaño/peso | Drafter | Por qué |
|---|---|---|---|
| `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` | 28B · ~18GB pesos NVFP4 (QAT) | `incoai/Qwen3.8-27B-DFlash2` (BF16 2B) | **Checkpoint de referencia del release** (77k descargas). QAT → calidad NVFP4 superior documentada (GSM8K 96.3, MMLU 83, HumanEval 100 con bf16-equiv). Debería arrancar la wheel (lm_head half). NOTA prod: NO es uncensored — sustituir coder390 cambia alineación/behavior (decisión de contenido, no solo técnica). |
| `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` | ~18GB (tag 18B) | ídem | **La variante que arregla nuestro bug**: mismo 27B NVFP4 pero lm_head BF16. Prueba A/B directa contra coder390 para aislar el bug y validar calidad antes de decidir checkpoint de prod en 1.5.2. |

## T2 — MoE de throughput (probada en esta familia de GPU)

| Modelo | Tamaño/peso | Drafter | Por qué |
|---|---|---|---|
| `Qwen3.6-35B-A3B` NVFP4 | 35B total / **3B activos** · ~19GB | `incoai/Qwen3.6-35B-A3B-DFlash2` (0.5B) + MTP4 (medido en #270) | Medida REAL en 4×V100 TP4: 174,76 tok/s con MTP4, GSM8K 95,3%. MoE = throughput por parámetro activo; nuestro rig ya sirvió qwen35moe (sprint 4, 2026-09). NVFP4-MoE TP2 con kernels indexados (#390). |

## T3 — Valida el stack GGUF nuevo (la gran novedad 1.5.2 / 1Cat-LLM)

| Modelo | Tamaño/peso | Drafter | Por qué |
|---|---|---|---|
| `QUASAR-QAT/gemma-4-12B-it-QUASAR-Q4_0-GGUF` | 12B · ~7GB | — | Gemma 4 con soporte SM70 (RMSNorm fusionada #1292); QAT con KL menor que el QAT oficial de Google. Ejercita el camino GGUF completo (loader nuevo, shards TP2 #1026). |
| `QUASAR-QAT/Muse-Glimmer-30B-QUASAR-Q4_0-GGUF` / NVFP4 | 30B · ~15-17GB | `incoai/Muse-Glimmer-30B-DFlash2-GGUF` (3B) | Multimodal (Image-Text-to-Text), familia nueva, con draft GGUF — demo de spec+GGUF juntos. |
| `QUASAR-QAT/Qwen3.5-4B-QUASAR-NVFP4` / `Q4_0-GGUF` | 4-5B · ~3GB | — | Smoke test barato del wheel: arranca en minutos, valida NVFP4 y GGUF de un tirón. (La variante W4A4 es de Blackwell — NO para V100.) |

## T4 — Frontera (ajuste justo o desconocido)

| Modelo | Tamaño/peso | Por qué / riesgo |
|---|---|---|
| `RadixArk/glm47-flash-blockwise-fp8` | 31B FP8 bloque (~31GB) | GLM-4.7-Flash en FP8: cabe MUY justo en TP2 (KV e5m2, util 0.90). Flags block_fp8/turbomind existen en la wheel. Riesgo: OOM por poco margen; probar con max-model-len corto. |
| `DeepSeek-V4.1-Flash` GGUF IQ2_XXS | MoE grande (release: 4×V100 TP4 + 33% expertos en host) | En TP2 exigiría ~50 %+ offload a host → decode lento. Solo como experimento de capacidad (256K/host-KV), no como serving. |
| `incoai/Qwen3.8-27B-Splash` | 27B | Variante nueva de spec-decode (Splash) — curiosidad, sin docs de rendimiento. |

## NO viables en 2×V100 (por tamaño)

- `RadixArk/Qwen3.8-Flash-Next-NVFP4` (**120B**) — el release lo sirve en 4×V100; en 64GB no cabe ni con host-KV. Su DFlash2 (0.6B) existe para el día que tengamos más GPUs.
- `RadixArk/GLM-5.3-Flash-NVFP4` (**168B**, medido en 8×V100) · `GLM-5.3-NVFP4` (381B) · `DeepSeek-V4-Flash` (8×V100 TP8/PP2×TP4) · Kimi-K3.1 · MiniMax-M3.1 · Ornith-1.5-122B.

## Orden de prueba sugerido (una ventana por pato)

1. **QUASAR 27B + DFlash2** (T1) — desbloquea la wheel + experience completa del release (spec, prefix-cache, c8). Si benchmarka bien → decisión de checkpoint de prod (¡ojo uncensored!).
2. **BF16-LMHead 27B** (T1b) — aísla el bug y da la alternativa directa a coder390.
3. **Qwen3.6-35B-A3B + DFlash2/MTP4** (T2) — la jugada de throughput MoE.
4. **Gemma-4-12B GGUF + Qwen3.5-4B** (T3) — valida el stack GGUF barato.
5. **Muse-Glimmer-30B + DFlash2-GGUF** (T3b) — multimodal + spec GGUF.
6. **glm47-flash FP8** (T4) — solo si sobra ventana.

Descargas T1-T3: ~18+4+19+7+3 GB ≈ 51GB (cabe en /srv con holgura).


## ANEXO 09-nov (II): deep-research de los 5 modelos pedidos por Joker

Soporte GGUF verificado EN la wheel v1.5.2 (grep local): loader + adapters `qwen35` (dense), `qwen35moe` (MoE), `qwen4exp`; cuants soportados por los kernels SM70 GGUF: **Q4_K_M, Q4_0, Q8_0, IQ4_XS/NL, IQ3_S/XXS, IQ2_***; **APEX NO soportado**; MTP-en-GGUF sin manejo explícito (el GGUF+MTP4 fue build dev #940, no release).

| # | Modelo | Arch GGUF | Cuant disponible | ¿Cabe TP2? | Veredicto |
|---|---|---|---|---|---|
| 1 | `IsValorum/Qwen3.8-35B-A3B-Distill-...APEX-I-MiniPlus-V2.1-Abliterated-GGUF` | qwen35moe ✓ | **SOLO APEX** (14,7GB, único fichero) + mtp-Q4_0/Q8_0 drafts | Sí (14,7GB) | ⚠️ **Probable rechazo**: cuant APEX no soportado por los kernels; MoE-GGUF además sin cualificar completa en el release. Con mtp-drafts incluidos (sin uso verificado). 104k descargas. |
| 2 | `bartowski/Jackrong_Qwopus3.8-27B-Coder-Flash-GGUF` | qwen35 ✓ dense (adapter CUALIFICADO) | **menú completo**: Q4_K_M (~16,5GB) ✓, Q4_0 ✓, IQ4_XS ✓, hasta bf16 | Sí | ✅ **VIABLE — top candidate**: coder + vision (mmproj) + tool-use/agents, fresísimo (07-nov). Text-only con limit-mm o con visión activada. |
| 3 | `IsValorum/Swift-1.5-Qwen3.8-27B-Uncensored-...APEX-I-MiniPlus-V3-GGUF` | qwen35 ✓ | **SOLO APEX** (14,2GB, único fichero) + mmproj-Q8_0 | Sí | ⚠️ Mismo problema de cuant APEX → probable rechazo. Uncensored con MTP en el base (GGUF sin MTP verificable). |
| 4 | `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-...NEO-CODER-MAX-MTP-GGUF` | qwen35 ✓ dense | **menú completo + quants MTP dedicados**: `-MTP-Q4_K_M` (~17GB), `-Q4_K_M`, `-MTP-Q8_0`, IQ4_XS… | Sí | ✅ **VIABLE — top candidate**: uncensored heretic, coder+creative+roleplay, multi-stage tuned, 1,5M descargas. Dos modalidades: MTP integrado (soporte en release wheel SIN verificar) o target-only garantizado. |
| 5 | `BAAI/AREX-2` | — (safetensors BF16, qwen3_5) | BF16 27,36B = **54,7GB** — sin GGUF disponible | ❌ NO cabe (64GB totales menos KV) | ❌ **Descartado hoy**. Modelo agente/deep-research interesante — vigilar si sale GGUF comunitario (un Q4_K_M ~15GB sería viable). |

### Plan de prueba recomendado (ventanas sobre la wheel venv152)

1. **DavidAU `-MTP-Q4_K_M`** (si MTP-GGUF no carga → `-Q4_K_M` target-only): el más alineado con el perfil de prod (uncensored coder + creative).
2. **Qwopus3.8 Coder-Flash `Q4_K_M`**: coder fresco con visión; probar text-only primero.
3. **Los dos APEX de IsValorum**: un solo boot de verificación cada uno para documentar el rechazo (y pedir en su repo quants IQ4_XS/Q4_K_M, que sí están soportados).
4. Bonus: todos son familia Qwen3.8-27B → el draft `incoai/Qwen3.8-27B-DFlash2-GGUF` es compatible como speculative draft (soporte GGUF-draft en release sin verificar — probar tras target-only OK).
