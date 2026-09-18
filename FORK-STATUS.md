# Estado del fork — SabaTech-dev/1Cat-vLLM

> Última actualización: 2026-09-18 (sync con upstream + clasificación de las 78 branches)

## SYNC MAIN ↔ UPSTREAM (2026-09-18) — HECHO

- main sincronizada con upstream/main (b711d5304525) vía merge: **nuevo sha 201dfad406e0794b76c9bbc7a86382e97824ab3a**.
- Estado vs upstream: **0 detrás / 3 adelante** (2 commits de docs + 1 commit de merge). FORK-STATUS.md preservado.
- La API `merge-upstream` devolvió 404 pese a fork registrado; se hizo vía clone mirror + merge + push.

## NUESTRO TRABAJO EN ESTE FORK (clasificación de las 78 branches, 2026-09-18)

Análisis con `git cherry` contra upstream/main (equivalencia de parche). Excluye `main`:
77 branches → **61 ya en upstream / 11 con trabajo único / 5 obsoletas**.

### Clase (ii) — trabajo único valioso, ranking por valor de mejora SM70/Volta

1. **sprint/f1-triton-attn** — 47 commits (+3875 líneas): backend propio **TRITON_PAGED**
   (paged-attention Triton para Volta), KV cache int8/int4 por cabeza, tiles QSA pre-Ampere
   (issues #441/#469), port del KV watermark (#44594), CUDA graphs decode-only, LIFO
   free-block reuse, hooks GDN warmup + INT8_CLIP. Incluye su propio disable de all-reduce.
2. **agent/private-v100-dsv4-deterministic-fp8-tactics-20260827** — bugfixes de acumulación
   FP8/MXFP4 en DeepSeek V4 (preservar acumulación por defecto, rutas decode no probadas off).
3. **agent/v100-dflash2-labd-chain-20260828-031506** — estabiliza cadenas DFlash2 en TP4,
   verificador exacto q32, cadenas drafter-free + resultados.
4. **agent/private-v100-dsv4-fused-fp16-aux-20260827** — revisión del kernel de fusión
   auxiliar FP16 exacta para V4.
5. **agent/v100-dsv4-pp2tp4-fused-fp16-aux-20260826** — kernel original de fusión de GEMVs
   auxiliares FP16 V4 + docs de screens rechazados (probablemente subsumido por el #4).
6. **codex/v100-glm53-fp16-quality-perf-20260828-040813** — preserva acumulación NVFP4 GLM
   en `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`.
7. **codex/v100-glm53-nvfp4-quality-guard-20260828-135709** — restaura quality guard NVFP4
   GLM (mismo fichero que el #6: mergear en ese orden y resolver solape).
8. **codex/v100-qwen38-flash-next-prefill-20260827-000848** — sube occupancy de
   selected-attention Qwen3.8 (`vllm/models/qwen4_exp/nvidia/ops/qsa.py`).
9. **contrib/restore-sm70-build + contrib/sm70-disable-custom-allreduce** — el MISMO commit
   (65dcec21f): disable custom all-reduce en SM70/Volta; es el contenido restante de #431 y #435.
10. **docs/adoption-plan-20260901** — solo documentación (roadmap F3–F7, 809 líneas, 4 ficheros).

### Clase (iii) — obsoletas (se dejan en sitio, sin borrar)

- codex/v100-dflash2-fp8-verify20-20260828-130320 (solo doc de diseño, campaña superada)
- dependabot/pip/chardet-7.6.0, datasets-5.0.1, minor-update-61b22425ee,
  prometheus-fastapi-instrumentator (bumps de dependencias ya superados por upstream)

## PRs ABIERTOS AL UPSTREAM (estado sobre upstream main actual, 2026-09-18)

- **#435** (restore SM70 compilation, 3 csrc fixes): OPEN, **mergeable=MERGEABLE**,
  mergeState=**UNSTABLE** (checks sin pasar). Los 3 fixes de csrc (cumem_allocator
  fabric-guard, etc.) **ya están absorbidos en upstream main**; el diff restante del PR es
  únicamente el parche de all-reduce, idéntico al de #431.
- **#431** (disable custom all-reduce SM70 TP2): OPEN, **mergeable=MERGEABLE**,
  mergeState=**UNSTABLE**. El parche NO existe en upstream (`use_custom_allreduce()` sigue
  devolviendo True incondicional). OJO: upstream acaba de activar rutas all-reduce en SM70
  TP4 (#605 perf/sm70-tp4-allreduce-defaults) — validar dirección antes de insistir.
- sprint/f1-triton-attn lleva un disable de all-reduce propio (commit aparte, mismo efecto):
  al mergearlo, el parche de #431 queda cubierto.

## PLAN DE MERGE RECOMENDADO (fork → producción; nada mergado aún)

1. `sprint/f1-triton-attn` (el de mayor valor; cubre también el disable de all-reduce).
2. `agent/private-v100-dsv4-deterministic-fp8-tactics-20260827`.
3. `agent/v100-dflash2-labd-chain-20260828-031506`.
4. `agent/private-v100-dsv4-fused-fp16-aux-20260827` → verificar si el 26 aún aporta.
5. `codex/v100-glm53-fp16-quality-perf` → después `codex/v100-glm53-nvfp4-quality-guard`
   (mismo fichero csrc, resolver solape).
6. `codex/v100-qwen38-flash-next-prefill-20260827-000848`.
7. `docs/adoption-plan-20260901` (solo docs).
8. Upstream: decidir #431 con maintainer (el parche quedará ya integrado en nuestro árbol vía
   sprint; #435 está absorbido salvo el all-reduce — mismo destino que #431).

## Posición de serving (2026-09-18)

- Prod: v1.5.0 (último release), denso, health estable.
- MoE híbrido vía vLLM: upstream main lo sirve YA (medido: single 91.2/101.0, 6-stream 392.3
  agregados, ctx 32K TP2) — build desde main el día que se active.
- Upstream sigue avanzando fuerte en SM70 (E4M3 KV, Q8192 prefill, Mamba grid #645, FP8 KV
  graphs B32) — revisar solapes con clase (ii) antes de cada merge.
- Benchmarks completos: repo sabatech-sm70-qflash, ROADMAP.md sección SERVING STACK.
