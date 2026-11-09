# Análisis RELEASE 1Cat-vLLM 1.5.2 — 08-nov-2026

> Release oficial: 06-nov-2026 08:50 UTC · 128 PRs · 1.805 commits · ~400k líneas nuevas.
> Wheel: `1cat_vllm-1.5.2+cu128-cp312-cp312-linux_x86_64.whl` (269,5 MB) · SHA256
> `a27afc512cf29d4baa322366f92b32f1d0ddec6d7178b0bf4351ad00c1958baa` — verificado.
> Construida desde main `9f970d628` (el siguiente commit `2bb8c2b9c` solo toca Docker).
> Fuentes: RELEASE_NOTES.en.md (archivada en `/srv/llm-stack/bench/upstream-v152-release-notes.md`),
> PR_INDEX.md, `git log v1.5.1..v1.5.2`, grep del árbol de la tag.

## 1. Qué promete la release (rig oficial: 4×V100-SXM2 NVLink, TP4, 350 W)

| Métrica | 27B NVFP4 + DFlash2 | Flash-Next NVFP4 + MTP4 |
|---|---|---|
| Decode single (8K ctx) | 487 tok/s (aceptación 70 %) | 253 tok/s |
| Throughput 8 concurrentes | 1224 tok/s agregado | 659 tok/s |
| Single a 250K ctx | 268 tok/s (−47 % vs 1K) | 213 tok/s (**−18 %**, host-memory KV) |
| HumanEval single | 619 tok/s (aceptación real 84 %) | 311 tok/s (91 %) |
| Prefill 8K | 4646 tok/s | 6289 tok/s |
| Primer token con 250K de historial | 1,7 s | 2,1 s |

- Contexto máximo 256K, concurrencia oficial 8 (limitación conocida #4: más concurrencia exige menos contexto).
- Calidad NVFP4 vs BF16: GSM8K/MMLU/HumanEval dentro del error estadístico; 0 salidas corruptas; camino de precisión transparente (FP4→FP16 una sola vez, GEMM FP32, logits FP32, estado GDN FP32, **KV FP8 E4M3 con calibración por capa**).
- Validación pre-release: matriz completa, 45+38 checks de API, runs de 20 min (~2900 requests) sin errores ni fugas de memoria, distribución de sampling contrastada contra greedy sin draft.

**Caveat para nuestro rig**: los números son TP4 NVLink a 350 W; nosotros somos 2×V100 PCIe TP2 con techo 200 W — no comparables directamente (referencia nuestra: Modo A conc16 = 216,6 t/s agregado medido).

## 2. Novedades por área (los 128 PRs agrupados)

### Spec-decode: de single-request a 8 concurrentes
- DFlash2 multi-request: proyecciones FP8 del draft compartidas entre 2-4 peticiones (#1289), verificación GDN BV4 para 2-3 peticiones (#1291), all-reduce TP4 + RMSNorm fusionados (#1292).
- Kernels para 9-32 filas de verificación (#1367, #1368).
- Sampling sin arrastrar: grafo del sampler cubre 1-4 peticiones (coste extra de sampling a c2/c4: ~3 ms → ~0,9 ms, #1393); top-k/top-p clasificados en device sin sync (#1267); **el historial de tokens ya no se reenvía al model runner cada paso** (#1263).
- Flash-Next MTP4 verificación concurrente (#1369); perfil release 27B sirve 8 (#1392).

### Contexto largo y prefill
- Chunks de prefill de 4096 tokens con atención Q4096 nativa a 79 TFLOP/s y workspaces reservados (#1287).
- Split-KV para chunks cortos tras prefijos largos (#1298); el prefill avanza en chunks completos junto a decoders residentes (#1295).
- Scheduling de paso mixto: CUDA graphs piecewise para prefills mixtos cortos (#1290); el coste fijo del paso mixto ya no se carga al presupuesto de prefill (#1288).
- **#1394: se acabó compilar kernels por cada longitud de prompt nueva durante el serving** (elimina stalls de first-token y crecimiento de memoria del worker).

### Caché de prefijo y agentes
- **Aciertos parciales de prefijo para modelos híbridos Mamba/GDN** (#1297) con bloques de estado de 2048 tokens (#1296); los turnos de agent reanudan en el último mensaje.
- Los bloques de salida de modelos con reasoning se cachean → multi-turno con thinking acierta caché (#1338).
- La API informa `prompt_tokens` cacheados (#1299). Correcciones de identidad de imagen en claves parciales (#1363) y de KV offload/EAGLE/MTP (#1360, #1304, #1244, #1240, #1265, #1232).

### Flash-Next: KV en memoria de host y PLE
- Historial FP8 KV en memoria de host; 256K/8-concurrentes es config por defecto del release; PLE disk-mapped automático, sin env var (#1392). PLE híbrido con pin (#1340), filas FP8 por fast lane nativa (#1343), prefetch/gather (#885, #996, #988).
- QSA (atención sparse de contexto largo): menos trabajo de selección (#1171), selección estable en batches mixtos (#817), RoPE clamped (#1201).

### Nuevo modelo: DeepSeek-V4.1-Flash en 4×V100
- IQ2_XXS GGUF desde fuente: 512K KV packed, Engram, DSpark5 spec-decode, ~33 % de expertos en host (#1277). Fixes DeepSeek-V4 (#1183, #1222, #715). Kernels GGUF (#1017, #993, #992, #1026, #1147).

### Estabilidad y memoria
- **#1391: el profiling de arranque del 27B sin draft en 256K cargaba 17,89 GiB de vision activations → ahora 1,18 GiB y la capacidad de KV se multiplica ×8.**
- Sizing/alineación de páginas KV híbridas (#1252, #1251, #1256, #1258); bloque KV nulo reservado al arrancar (#1247).
- MM: caches MRoPE acotadas (#1019), **la caché de proceso en shm ya no fija items para siempre** (#1200), EXIF/transparentes (#1366, #1310).
- Comunicación: **pynccl all-reduce out-of-place por defecto, restaura la aceptación especulativa** (#1284); checkpoints sharded incompletos fallan rápido (#1356); **las notificaciones de shm siguen polleando mientras los lectores están ocupados** (#1306).

### API y tool calls: 12 fixes de corrección
- Tool calls: XML branch preservada con response_format JSON (#1312); streaming de tool calls/errores/logprobs preservado (#1317); enteros grandes (#1308); JSON válido con no-finitos (#1309); herramientas sin parámetros (#1315); `parallel_tool_calls=false` se cumple (#1316); `tool_choice="none"` en streaming (#1314); qwen3_coder completa el body tras el header (#1313).
- Salida: stop strings alineadas (#1305), logprobs de prompt en DELTA paralelo (#1318), ventana pendiente del detokenizer acotada (#1354), salida sin logprobs si las filas no casan (#1357).

### Compilación, build y plataformas
- torch.compile: la eliminación de clones ya no cambia layouts ni crea aliases (#1358, #1359); los límites multimodales entran en la clave de la compile-cache (#1355).
- Turing SM75: FA2 por worker (#797), QPN/QSA indexer (#804, #1229), metadata GDN especulativa fusionada (#1264), DeepSeek-V4 (#798). SM72 targets (#803).
- Docker: imagen SM70 de release con los dos modelos y el compilador CUDA 12.8 incluido (#1392, #1395).

## 3. Impacto directo en nuestros bloqueos de campaña

| Bloqueo nostro | Estado en v1.5.2 |
|---|---|
| **E4M3 rechazado** (XQA `tmp_out must be fp32`) | **DESBLOQUEADO**: el grep de la constraint en el árbol v1.5.2 está vacío y E4M3 es el camino KV por defecto de la release (calibración por capa). Era nuestro mayor ensanche de calidad/capacidad bloqueado. |
| Familia #865 (shm_broadcast / stall spec ≥2 concurrent) | **SIN fix nominado**. La issue sigue OPEN en 1CatAI (solo nuestro comentario del 05-nov). Toques adyacentes: #1306 (polling shm), #1284 (all-reduce out-of-place), rework completo de concurrencia spec (#1289-#1292, #1367-#1368, #1393). Revalidar en nuestro rig antes de re-activar spec. |
| Sprint-fix chunked FC projection (OOM de drafter bajo prefill concurrente, `30dd61ae1`) | **NO incluido** (no es ancestro de la tag). La línea oficial puede conservar ese OOM de DFlash2. |
| DFlash2 corrupto con T>0 en main post-v1.5.1 | Pendiente de verificación en el retest (no hay PR nominado en las notas). |
| Retest v1.5.2 (antes bloqueado por "sin tag" y regresión de memoria) | **Desbloqueado**: existe la tag y el wheel oficial; #1391 reduce el profiling de arranque ×15 y #1394 elimina el growth de memoria por prompt-length (dos causas plausibles de los 98 OOMs históricos del retest). |
| #652 (deadlock custom_all_reduce TP2) | Cerrada oficialmente el 31-oct (compilar desde fuente) — familia vecina, contexto útil para #865. |

## 4. Topología de imports del serve :8009 (verificada hoy, importante)

- El proceso vivo de prod importa el **árbol sprint** `/srv/benchmarks/lhu/upstream/1cat-vllm` (`v1.5.1.dev91+g30dd61ae1`, rama `sprint/f1-dflash2-chunked-cast`) + `/srv/build-1cat/src/flash-attention-v100`, vía **`PYTHONPATH` del env file (línea 6)**.
- El venv `/srv/benchmarks/1cat` tiene un editable del **29-oct** (`1.5.2.dev1768+g03485453b.cu129`) apuntando al worktree `/home/joker/repos/1Cat-vLLM-worktrees/campaign-upstream-main` — **sombreado por el PYTHONPATH**. Solo tomaría el control si el PYTHONPATH desapareciera (mina latente; documentada).
- Consecuencia operacional: el switch oficial a 1.5.2 debe **cambiar el venv y QUITAR el PYTHONPATH explícitamente en el mismo drop-in**, nunca solo una de las dos cosas.

## 5. Actualización ejecutada hoy (sin tocar prod)

1. Wheel descargado del release oficial + SHA256 verido contra `SHA256SUMS` ✓.
2. Archivado en `/srv/llm-stack/bench/wheels/` (`1cat_vllm-1.5.2+cu128-...whl` + `SHA256SUMS-1.5.2`).
3. **Venv paralelo** `/srv/benchmarks/1cat/venv152` (uv, python 3.12): wheel + torch 2.10.0+cu128 (pinned, misma versión que el venv actual) + flashinfer 0.6.11.post2 + cubin.
4. Sanity CPU (CUDA oculto, prod intacta): `import vllm` → **`1.5.2`** ✓; `vllm serve --help` funciona y autodetecta `CUDA_HOME=/usr/local/cuda` ✓; launchers `serve_qwen38_27b_nvfp4_v100.sh` (perfil release: DFlash2 pinned, c8, util 0,83) y `serve_flash_next_nvfp4_v100.sh` presentes.
5. Requisito del wheel (JIT de kernels GDN prefill de FlashQLA al primer uso): **nvcc 12.9** en `/usr/local/cuda` ✓ (≥12.8 requerido).
6. Prod :8009 intacta: boot 06-nov 19:25 (uptime 2d+), `speculative_config=None`, KV 1.083.582 tokens, 0 fatal errors, health 200. **Vigilancia 48 h de la clase #865 SUPERADA** (0 crashes desde el 05-nov 22:40 en Modo Throughput).

## 6. Plan de switch de prod (PENDIENTE de ventana — decisión del dueño)

Alcance: un drop-in nuevo (p. ej. `zz-zzzzzzz-v152.conf`) que:
- `ExecStart=` → `/srv/benchmarks/1cat/venv152/bin/vllm serve ...` con la **misma config Modo A** (max_num_seqs 16, prefill 8192, e5m2 o E4M3, limit-mm 0) — probando E4M3 AHORA es viable.
- `Environment=PYTHONPATH=` (vacío, para que el wheel no quede sombreado por el árbol lhu).
- Validación: canario VL, `speculative_config=None`, KV size, probe TTFB, A/B spot-check; telemetría + watchdog armados.
- Rollback: desactivar el drop-in nuevo (rename `.disabled`) + `daemon-reload` + `restart` (el drop-in del Modo A actual queda como configuración de caída).
- Requisito de la ventana: prod parado ~6-8 min (compilación JIT del primer uso incluida).

Fuentes: release page 1CatAI/1Cat-vLLM v1.5.2 · notas EN + PR_INDEX (archivados) · árbol git de la tag (fetch upstream en `/home/joker/repos/1Cat-vLLM`).


## ANEXO 09-nov: A/B ejecutado — la wheel NO puede servir nuestro checkpoint (veredicto)

A/B en ventana GPU (BASE medido x2 + 4 intentos de la wheel con gates):
- **BASE (sprint v1.5.1.dev91+g30dd61ae1, e5m2)**: single 256tok 44,7-44,8 t/s · multi6 207,5-212,1 agregado · prefill ~1000/~5000/~8300 t/s @391/@2065/@5467 · sonda tg128 c1/c4/c8/c16 = 39,0/108,9/161,2/224,2 y 38,9/110,4/164,0/213,8 (repro <4%) · KV 1.083.582 · sniffs OK.
- **Wheel v1.5.2 + e5m2**: NO ARRANCA — crash determinista 6/6 intentos (30 min): `AttributeError: 'ParallelLMHead' object has no attribute '_layer_execution_policy'` en `linear.py:375 process_weights_after_loading` (camino ModelOpt NVFP4/TurboMind; asignacion de politica en linear.py:489 que ParallelLMHead no ejecuta).
- **Wheel + E4M3**: mismo error raiz (no era constraint XQA ni dtype — es el lm_head CUANTIZADO de coder390; el checkpoint de referencia Unsloth lo lleva en half y por eso el release no lo ejercita).
- **256K en wheel**: no testable (no arranca).

**Mecanismo**: el checkpoint de referencia del release (Unsloth) lleva lm_head en half; los checkpoints comunitarios con lm_head cuantizado crashean al proceso de pesos. Nuestro sprint (v1.5.1-era) arranca el mismo checkpoint sin problema.

**Veredicto del gate de upgrade**: FALLA el primer criterio (no sirve el workload) → **NO upgrade**. Prod permanece en sprint Modo A (48h+ validado). Draft del issue upstream: `/srv/qflash/upstream-152-boot-bug-draft-20261109.md` (pendiente GO para publicar).

**Rutas de desbloqueo** (I+D): (a) fix upstream del getattr/policy para lm_head cuantizado; (b) probar el checkpoint Unsloth de referencia en la wheel; (c) parche local de la wheel (medicion, no prod); (d) fork-merge release+sprint cuando exista fix.
