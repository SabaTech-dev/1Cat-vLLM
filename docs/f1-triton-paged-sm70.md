# Sprint F1 — Backend de atención paged Triton autocontenido para Volta SM70 (`TRITON_PAGED`)

Rama: `sprint/f1-triton-attn` (base: `contrib/restore-sm70-build`)
Fecha: 2026-09-02
Estado: esqueleto funcional + validación numérica en CPU (intérprete de Triton). Pendiente de ejecución en GPU.

## Objetivo

Disponer de un backend de atención para SM70 (V100) que NO dependa de
`flash_attn` ni FlashInfer: solo Triton y PyTorch. Porta el diseño de
referencia de nano-vllm / MinivLLM a la interfaz V1 de este fork y queda
registrado como `TRITON_PAGED` para poder seleccionarlo mediante
configuración y compararlo (A/B) contra `FLASH_ATTN_V100`.

Referencias estudiadas:

- nano-vllm (`GeeeekExplorer/nano-vllm`, `nanovllm/layers/attention.py`):
  kernel Triton `store_kvcache` de fichero único y delegación del resto en
  `flash_attn_varlen_func` / `flash_attn_with_kvcache`.
- MinivLLM (`Wenyueh/MinivLLM`, `src/myvllm/layers/attention.py`): la
  referencia clave — paged KV store en Triton, decode paged con block
  tables y prefill varlen flash, todo sin `flash_attn`.

## Qué se ha implementado

### Kernels — `vllm/v1/attention/ops/triton_paged_attn.py`

Los tres kernels operan sobre las vistas NHD del KV cache del fork
(`kv_cache.unbind(1)` sobre `[num_blocks, 2, block_size, num_kv_heads,
head_size]`), idénticas al layout de MinivLLM:

1. `store_kvcache_paged` — scatter store con `slot_mapping` plano
   (`slot = bloque_físico * block_size + offset`). Grid
   `(num_tokens, num_kv_heads)`; salta slots `-1` (padding).
2. `paged_decode_attention` — decode paged con block tables, softmax
   online en fp32 con el truco `exp2` (escala plegada con `log2(e)`),
   GQA mediante `head_kv = head_q // (H / KVH)`. El gather de la block
   table es por token, por lo que un chunk puede cruzar bloques físicos
   no contiguos. La fila de query se resuelve con `query_start_loc`.
3. `varlen_paged_prefill_attention` — prefill/extend varlen. TODO el KV
   (prefijo cacheado + tokens nuevos) se recupera por la block table:
   como en vLLM la actualización del KV cache corre ANTES de la
   atención, el kernel es uniforme para primer prefill, prefill
   troceado (chunked prefill) y prefix caching. La causalidad usa
   posiciones globales (`q_pos = context_len + offs_m`).

### Backend — `vllm/v1/attention/backends/triton_paged.py`

- `TritonPagedMetadataBuilder`: pass-through de
  `CommonAttentionMetadata`. Sin soporte CUDA Graph (negado
  explícitamente; ver «Qué queda»).
- `TritonPagedBackend`: shape de cache
  `(num_blocks, 2, block_size, num_kv_heads, head_size)`, layout
  requerido `NHD`, dtypes `fp16/fp32` (Volta no tiene BF16), KV dtype
  `auto/float16`, `block_size` múltiplo de 16, `head_size` potencia de
  2 en [16, 256], compute capability limitada a (7, 0).
- `TritonPagedImpl`: `forward` enruta por `max_query_len` (== 1 →
  decode; > 1 → prefill/extend, que también cubre decode multitoken de
  especulación). `do_kv_cache_update` invoca el kernel de store — el
  fork exige `forward_includes_kv_cache_update = False` en TODOS los
  backends V1.

### Registro y selección

- `vllm/v1/attention/backends/registry.py`: nuevo miembro
  `TRITON_PAGED = "vllm.v1.attention.backends.triton_paged.TritonPagedBackend"`.
- `vllm/platforms/cuda.py`: `TRITON_PAGED` se añade SOLO a la lista de
  prioridades de la rama SM70 + `VLLM_SM70_FLASH_ATTN_V100` (tras
  `FLASH_ATTN_V100`). Ningún comportamiento por defecto cambia: en auto,
  `FLASH_ATTN_V100` sigue ganando; `TRITON_PAGED` solo se elige si se
  selecciona explícitamente o si los previos son inválidos.

Selección explícita:

```bash
VLLM_SM70_FLASH_ATTN_V100=1 vllm serve <modelo> \
    --attention-backend TRITON_PAGED -dtype float16
```

(`AttentionConfig.backend` acepta el string y lo resuelve contra el
enum; `validate_configuration` rechaza con motivo explícito cualquier
combinación no soportada — BF16, fp8 KV, MLA, sink, otra CC.)

## Validación realizada (sin GPU)

`tests/v1/attention/test_triton_paged_kernels_cpu.py` ejecuta los tres
kernels con `TRITON_INTERPRET=1` (intérprete de Triton sobre CPU) y los
compara contra una referencia densa de PyTorch:

- store: roundtrip exacto.
- decode: diff máx. 2.4e-4 (GQA, seqs 1..64, bloques múltiples).
- prefill: diff máx. 2.0e-3 — primer prefill, chunked con prefijo
  cacheado, queries que ocupan varios tiles (33 tokens con BLOCK_M=32),
  head_dim 64 y 128, tolerancia 3e-2.

Durante este proceso se detectaron y corrigieron DOS bugs reales de
kernels que solo el intérprete hizo visibles sin GPU: la orientación
transpuesta del tile de V en el prefill y un broadcast de `m_new` por el
eje equivocado en `qk - m_new` (difuminaba el softmax por columnas en
lugar de por filas).

Ejecución (requiere pytest en el entorno):

```bash
TRITON_INTERPRET=1 python -m pytest \
    tests/v1/attention/test_triton_paged_kernels_cpu.py -v
```

## E2E en V100 (2026-09-02, cierre del punto 1)

Servido Qwen/Qwen3-0.6B fp16 (CUDA_VISIBLE_DEVICES=1, junto a la carga
productiva, ~2.6 GB) con `--attention-backend TRITON_PAGED`,
`--enforce-eager`, sin CUDA Graphs.

**Primer intento: salida corrupta desde el primer token.** Sonda
standalone con tensores contiguos: kernels correctos (prefill 3e-4).
Con las vistas `kv_cache.unbind(1)` que entrega el engine (no
contiguas, stride de bloque 2x): prefill max_diff 1.585 (basura).
Raiz: los kernels asumian caches contiguos (cero parametros de
stride). Fix en `a327d45b7` (strides explicitos en los 3 kernels).

**Post-fix**: greedy byte-identico a FLASH_ATTN_V100 en prompt corto
("La capital de Francia es París") y largo (72 tokens). Ambos
backends arrancaron junto a la carga productiva sin impacto.

## Qué queda (orden propuesto)

1. ~~**Prueba funcional end-to-end en V100**~~ (HECHA 2026-09-02: paridad greedy byte-identica vs FLASH_ATTN_V100 tras el fix de strides a327d45b7).
2. ~~**CUDA Graphs**~~ (RESTABLECIDOS 202-09-03: decode-only, greedy 20/20 identico a eager, decode 3.4-4.2x mas rapido. La "corrupcion" original era un falso positivo de canario — ver seccion.)
3. ~~**Autotuning**~~ (HECHO 2026-09-02: decode BN=128, prefill BM=16/BN=64; datos en f1-autotune.json). Queda opcional: partir el bucle KV en el bloque diagonal.
4. **Rendimiento decode**: el kernel decode es 1 fila por programa; si
   el batch pequeño deja la GPU ociosa, evaluar split-K (varios
   programas por secuencia + reducción) como `TRITON_ATTN`.
5. **Quality gates A/B vs FLASH_ATTN_V100**: misma batería que el
   experimento SM70 previo (logprob/perplexity sobre corpus fijo,
   greedy match). Criterio de promoción: paridad numérica < 1e-2 y
   throughput decode dentro del 10 % de FA-V100 antes de considerar el
   backend para nada más que experimentación.
6. **fp8 KV cache** (dequant en kernel) y sliding window si el caso de
   uso lo pide.
7. **Ampliar la gate de CC** a Turing/Ampere una vez validado en
   hardware (los kernels no contienen nada específico de SM70).

## Quality gates A/B y tuning (2026-09-02, cierre del punto 5 y 3)

GPU 1 dedicada (solo Qwen3-0.6B fp16 cargado, 32GB libres), offline
LLM API, 20 prompts fijos + corpus de ~6k tokens.

| Metrica | TRITON_PAGED | FLASH_ATTN_V100 | Delta |
|---|---|---|---|
| PPL media (corpus fijo) | 6.5771 | 6.5772 | 0.0001 |
| Greedy match (20 prompts) | 19/20 | - | 1 divergencia tardia sinonima |
| Logprob \|diff\| medio | 0.028 | - | ruido fp16 |
| Decode b1 (tok/s) | 23.3 | 20.0 | **+16.4%** |
| Decode b8 (tok/s) | 174.6 | 152.0 | **+14.9%** |
| Decode b16 (tok/s) | 330.7 | 308.4 | **+7.2%** |
| Prefill 2048 (ms) | 584.9 | 111.2 | 5.3x mas lento |

**Veredicto de promocion**: criterio cumplido (PPL diff 1e-4 < 1e-2;
decode no solo esta dentro del 10% sino que GANA en todos los batches).
Debilited documentado: prefill largo 3x mas lento que FA incluso tras
tuning (frontera Triton-vs-CUDA-hand-tuned en Volta).

**Tuning aplicado** (f1-autotune.json, mediana de 20 corridas CUDA
events): decode BLOCK_N 64->128 (-30% a 4k ctx); prefill BM 32->16
(2.6x kernel; BM>=64 colapsa por spilling de los 64KB smem de Volta).

## CUDA Graphs: RESTABLECIDOS (2026-09-03, re-evaluacion)

Historia de la correccion: el intento del 2026-09-02 se reverso por una
"corrupcion de la primera peticion" que resulto ser un FALSO POSITIVO —
Qwen3-0.6B es un modelo BASE y "el **Estados Unidos**..." es su
continuacion raw legitima de "La capital de Francia es" (el "Paris"
esperado venia del path con CHAT TEMPLATE del test de server E2E).
Verificacion final a util 0.85 dedicada:

- Greedy 20/20 IDENTICO graphs-vs-eager (bateria completa)
- PPL identica (6.5758 ambos)
- Decode: 4.2x (b1: 310 vs 73 tok/s), 3.8x (b8: 1946), 3.4x (b16: 3309)
- Prefill sin cambio (249 vs 250 ms; decode-only capture)
- piecewise y decode-only ambos correctos; se restaura como
  `UNIFORM_SINGLE_TOKEN_DECODE` (el mas conservador, rendimiento igual)

Leccion registrada: el canario de corrupcion debe corresponder al API
path del test (raw completion vs chat template); juzgar completions raw
de un modelo base contra expectativas con template produce falsos
positivos deterministas.

## Modelo de produccion en vLLM/Volta: infeasible (2026-09-02)

HauhauCS Qwen3.8-27B (twolven AWQ-MTP, compressed-tensors W4A16):
el scheme exige capability >= 7.5 (Turing); V100 = 7.0 -> no carga.
La alternativa local BF16 (~50 GB, qflash/hf) no cabe en 32 GB.
CONCLUSION para el plan de adopcion: servir la familia HauhauCS en
V100 requiere una cuantizacion compatible con SM70 (GPTQ con kernel
viejo o similar) o permanecer en llama.cpp. Los gates A/B y tuning
de F1 se validaron con Qwen3-0.6B fp16 (dedicada).

## Riesgos

- **Sin CUDA Graphs el decode pierde throughput** frente a FA-V100 en
  batch pequeño (lanzas Triton por capa y paso). Es el coste conocido
  del esqueleto.
- **`max_query_len == 1` como ruta decode**: en pasos con specs
  multimuestra la carga cae al kernel de prefill (correcto, pero más
  lento que un kernel decode dedicado multitoken).
- **El intérprete de Triton no es la GPU**: cubre semántica, no
  rendimiento ni comportamiento de shared memory/ptxas en SM70. La
  prueba en V100 es ineludible.
- **Prefill con prefijo cacheado no optimizado**: el kernel recorre
  TODO el KV desde el bloque 0; no hay salto de bloques previos al
  diagonal. Correcto, pero el coste es O(seq_len) por tile de query.

## Archivos tocados

| Archivo | Cambio |
| --- | --- |
| `vllm/v1/attention/ops/triton_paged_attn.py` | Nuevo. Kernels Triton (store, decode, prefill). |
| `vllm/v1/attention/backends/triton_paged.py` | Nuevo. Backend completo (builder, backend, impl). |
| `vllm/v1/attention/backends/registry.py` | Miembro `TRITON_PAGED` en `AttentionBackendEnum`. |
| `vllm/platforms/cuda.py` | `TRITON_PAGED` en prioridades SM70 (rama `VLLM_SM70_FLASH_ATTN_V100`). |
| `tests/v1/attention/test_triton_paged_kernels_cpu.py` | Nuevo. Tests numéricos CPU vía intérprete. |
| `docs/f1-triton-paged-sm70.md` | Este documento. |
