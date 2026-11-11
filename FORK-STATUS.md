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

> **Regla de ubicación de docs (corrección Joker 09-nov):** TODO lo referente a
> 1Cat-vLLM (releases, A/B, issues upstream, benchmarks del motor) se documenta en
> ESTE fork (`/srv/build-1cat/src-v151`, rama de campaña) — NO en `/srv/qflash`,
> que es el repo de `SabaTech-dev/sabatech-sm70-qflash` (campaña del motor qflash).
> Qflash conserva sus logs rodantes (CHANGELOG/ROADMAP/FORK-STATUS) como registro
> de campaña, pero los documentos temáticos 1Cat viven aquí.

---

## BARRIDO UPSTREAM — SWEEP 2026-11-11 (ventana 08-nov → 11-nov)

> Corrección de premisa: el checkout de docs de esta campaña es `/srv/build-1cat/src-v151`
> (rama `retest/v1.5.1-fork-merge`), no `/srv/qflash` (campaña anterior). Los bloques
> de sweep de aquí en adelante viven en este fichero.

### Nuestros PRs/issues upstream (estado real 11-nov)

| Ref | Estado | Movimiento |
|---|---|---|
| PR #1322 (fix build SM70) | **MERGED 10-nov** (merge `9b6f88da16`, yangzhuxinyzx lo reparó y aprobó) | el fix build SM70 está en upstream main |
| PR #431 / #435 | CLOSED 02-oct | contenido absorbido por upstream; FORK-STATUS de 18-sep quedaba desactualizado |
| Issue #1465 (1.5.2 wheel TP1/TP2 crashes) | OPEN — **primer datapoint externo**: @gelim corre NVFP4+DFlash2 TP2 bien con el perfil del script (E4M3 KV); #1483 y #1424 (otros autores) también arrancan NVFP4 con `fp8_e4m3`. **Hipótesis refinada**: el crash `_layer_execution_policy` es específico de KV **e5m2** en rutas NVFP4 SM70 (las rutas cualificadas son E4M3, #1424). El crash GGUF `_apply_gguf_embedding` sigue siendo defecto aparte del loader. Triage publicado (comentario 6806493403): re-test NVFP4 con e4m3 pendiente (re-download radixark 23GB) | |
| Issue #433 (build SM70 roto en main) | OPEN pero **cerrable**: arreglado por #1322 — comentario propuesto (6806493785) | |
| Issues #505 / #488 / #430 / #1220 | OPEN, sin movimiento desde sep/oct (stale) — candidatas a re-triage o cierre cuando la 1.5.x se estabilice en SM70 | |

### Actividad upstream desde 08-nov (post-release v1.5.2 del 06-nov)

- **#1478 "qwen-tp-unified" MERGED 10-nov** — ola de commits SM70 Qwen TP/MTP: strided GDN
  output storage, budget de draft candidates, QK fusionado entre head shards, collective
  fusion en MTP shards, particiones dense head, auditoría de prefill scheduling/PLE.
  **RELEVANTE**: es la línea exacta de nuestro prod (Qwen3.8-27B TP2) — un build
  post-#1478 puede cambiar el rendimiento de prod; candidato a re-bench.
- **#1482 MERGED 11-nov**: DeepSeek-V4.1-Flash nativo PP4×TP4 en 16×V100 — WATCH (no aplica
  a 2×V100 directamente, pero confirma inversión upstream activa en SM70).
- Issues nuevas: **#1483** (TP2 KV pool exhaustion mata EngineCore en vez de encolar —
  RELEVANTE a prod: nuestra KV es 1.08M y servimos concurrentes largos), #1480 FlashStates
  (WATCH), #1423 TORCHINDUCTOR_CACHE_DIR=/tmp (WATCH menor), #1424 (rutas E4M3 cualificadas
  solo hasta 262K; Yarn 1M falla — RELEVANTE si jugamos ventanas Yarn).
- Release vigente: v1.5.2 (06-nov). No hay v1.5.3 todavía.

### Otros repos vigilados

- NVlabs/cutile-rs: sin releases en ventana (última v0.4.0, 25-sep) — NULO.
- PrismML/llama.cpp: API de releases 404 — **premise correction**: el repo/org exacto hay
  que re-verificarlo (el dato de FORK-STATUS era de la era qflash).
- signalnine/q27: push 05-nov, FUERA de ventana (08-nov→11-nov) — NULO para este sweep.
- arXiv: NULO ×2 queries (toolchain opción 1 disponible: plugin `research_papers` en sesión;
  queries "speculative decoding KV cache quantization fp8" strict/mes y amplia — 0 resultados).

### Impacto en prioridades

1. **Re-test #1465 con E4M3** (re-download radixark NVFP4 23GB + flags mínimos vs prod) —
   puede desbloquear NVFP4 en TP2 y refinar el triage upstream. ALTA.
2. **Re-bench de prod contra un build post-#1478** (ola Qwen TP unified) — posible ganancia
   decode TP2 sin tocar flags. ALTA (requiere build propio de main).
3. **A/B KV e4m3 vs e5m2 en el sprint prod** (mismo tamaño de KV, rutas cualificadas,
   mejor mantisa) — MEDIA, dependiente de 1 y 2.
4. Limpieza de issues stale propias (#505/#488/#430/#1220) cuando 1-2 aterricen. BAJA.
