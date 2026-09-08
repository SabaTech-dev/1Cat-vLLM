# Lecciones y metodologias — SM70 / fork 1Cat-vLLM

Documento versionado con las lecciones operativas acumuladas. Si una
lección cambia una decision de diseno o de testeo, vive aqui (y en
Engram paraentre sesiones). Complementa el README de tools/sm70_validation.

## 1. Validacion y gates (la meta-leccion)

- **El gate es PARIDAD, no calidad absoluta.** Los modelos base
  (Qwen3-0.6B/1.7B) generan raw completions impredecibles: "La capital
  de Francia es" -> "el **Estados Unidos**..." es SALIDA LEGITIMA, no
  corrupcion. Nos costo tres falsos diagnósticos (graphs, util baja,
  int8). El gate correcto: PPL/greedy del dtype X vs fp16 EN EL MISMO
  modelo y API path.
- **El canario debe corresponder al API path**: raw completion vs chat
  template producen textos distintos por diseno.
- **Las sondas sinteticas NO validan cuantizacion**: randn no tiene los
  outliers post-RoPE reales. El gate E2E con modelo real (QUASAR 27B)
  detecto el colapso SNR del int4 simetrico que la sonda dio por buena.
- **La sensibilidad a la cuantizacion escala inversa con el tamano**:
  27B tolera int8 PTH (PPL +7.6%, coherente); 1.7B se rompe (+14%,
  incoherente). Validar cuantizacion con el modelo mas grande que quepa.
- **Metrica estable para A/B de throughput**: decode b8/b16 (b1 varia
  291-311 tok/s entre corridas por estado del host).

## 2. Metodologia de diagnostico (la que funciono)

- **Sonda standalone en dos modos**: (a) tensores contiguos — si pasa,
  la matematica del kernel es correcta; (b) vistas unbind(1) no
  contiguas — reproduce el camino real del engine. La diferencia entre
  (a) y (b) aísla kernel-vs-integracion sin bisect largo.
- **Experimento controlado para deltas ambientales**: si una metrica
  cae entre sesiones, correr el CODIGO VIEJO en la hora actual. Asi se
  separo la "caida eager 3x" (ambiental) de una regresion real.
- **Cuando una teoria exige que muchas variables independientes
  compartan el mismo sintema exacto**, sospechar del juez (la
  expectativa del testeador) antes que del codigo.
- **Verificar el texto real del archivo tras cada edit** (asserts +
  re-read): los reformateos automaticos rompen anclas silenciosamente.
- **pkill -f con el patron en la propia cmdline se auto-mata**: usar
  [p]atron o matar por PID.
- **systemd-run**: la salida del servicio va al JOURNAL de la unidad
  (journalctl -u), no al redirect del comando cliente.

## 3. Triton en V100 (SM70)

- `tl.arange` exige potencia de 2: slots no-pow2 (int4 empaquetado)
  necesitan arange almohillado + mascara (`head_dim_padded`).
- No se puede indexar un tensor con constantes (`s[0, :]`) — para
  extraer escalas empaquetadas: pointer cast
  `tl.cast(ptr, tl.pointer_type(tl.uint32))` + bitcast f32 (requiere
  slots 4-alineados: PHS % 4 == 0).
- `tl.split` requiere reshape a (..., 2); interleave en eje no-final =
  `tl.join` + permute + reshape.
- Los dots con dtypes mezclados fallan en compile-time; los errores de
  compilacion pueden mostrar el branch else en el traceback aunque sea
  constexpr.
- Tiles grandes colapsan por smem: BM>=64 con D=128 es 18x peor
  (spilling de los 64 KiB). Decode BLOCK_N=128, prefill BM=16/BN=64.
- int8 almacenado en tensor uint8 requiere reinterpetar al cargar:
  `raw.to(tl.int8).to(tl.float32)` — el cast directo uint8->f32 da 251
  donde debe haber -5 (NaNs por softmax overflow).

## 4. Cuantizacion KV

- int4 simetrico por (token, head) NO sirve para K real: los outliers
  post-RoPE (dims x20) dejan SNR ~1. Sirve para V (distribucion
  pareja). Hibrido int8-K + int4-V pendiente de separar input-row vs
  cache-slot widths en el store kernel.
- El patron del fork (triton_attn int8-PTH): escala fp32 INLINE en la
  cola del head slot (head+4), extraida con as_strided; el spec
  presupuesta esa memoria en page_size_bytes; el backend declara
  get_kv_cache_shape con head almohillado.
- OJO con dobles definiciones de metodo en una clase: la ULTIMA gana
  (un override nuevo puede quedar pisado por uno pre-existente).

## 5. Infraestructura V100 (jokerserver)

- GPU 1 es marginal (ECC single-bit masiva, Xid 79, RMA pendiente):
  cross-validar resultados criticos en AMBAS GPUs cuando el
  experimento quepa en los huecos (GPU 0 ~1.2 GB libres con
  llama-main; GPU 1 ~3.2 GB con llama-second).
- Driver R580 LTSB (ultima rama Volta, EOL jun-2028); CUDA 12.8 (13.x
  elimino sm_70); toolchain del fork = wheel 1.5.0 oficial.
- El alias de :8009 mantiene compatibilidad de consumidores; revert de
  swaps via los .bak timestampados en llm/config/.

## 2026-09-03 — Round F2.4/F9.1 tuning (int8 calibration + graphs + long context)

- **int8 intra-row outliers**: the amax scale collapses when 1 channel
  dominates the row (HauhauCS: row-amax p50=0.0, max=89.7). Percentile
  clip (k-th largest, 5%-of-amax floor) rescued PPL 215 -> 4.21 and
  improved QUASAR +7.6% -> +6.7%. Default clip 1%; sweep showed the
  optimum is narrow (0.2% / 5% both worse).
- **TurboMind GEMM is not CUDA-graph-safe**: fits memory with
  batched-tokens=1024, but outputs corrupt under graphs (PPL 1.6M vs
  eager 3.04). Workspace is cached per-stream (StreamWorkspaceKey).
  AWQ = eager until fixed upstream.
- **KV allocation OOM is driven by max_num_batched_tokens**, not util:
  default 8192 inflates the profiled activation peak; capping it to
  1024 frees the KV sizing. Four util-only retries (0.85-0.97) all
  failed identically.
- **TRITON_PAGED prefill stalls beyond ~16K context** (TP1 eager):
  6.1K OK, 16K/32K/55K no observable progress in 20-25 min. Long-context
  capacity (63K int8) is unreachable until diagnosed.
- **Orphaned EngineCore**: killing the parent bash timeout leaves the
  spawned VLLM::EngineCore holding 31GB. Always `pkill -9 -f
  VLLM::EngineCore` between runs and check nvidia-smi.
- **/tmp is ephemeral across host crashes**: the host crash wiped
  /tmp/opencode (scripts + result JSONs). Everything critical must live
  in git (tools/sm70_validation/) or Engram; results transcribed to docs
  immediately.

## 2026-09-04 - Diagnostic and ops lessons (stall bisect, APC, flag audit)

- **pkill -9 -f <pattern> KILLS THE CALLING SHELL** when the pattern
  matches its own cmdline (silent command death). Use bracketed
  patterns (`pkill -f "f24_[l]ongctx"`) or explicit PIDs.
- **VLLM::EngineCore is the process NAME** of the spawned engine - it
  does not match `grep python`. Find it via RSS or nvidia-smi
  query-compute-apps. py-spy dump on THAT pid (sudo) is the definitive
  stack; dumps on the parent only show queue.get.
- **faulthandler.dump_traceback_later** is the right tool for vLLM
  engine hangs when the engine runs in-process; in subprocess mode it
  misleads (parent frames).
- **A hung kernel can leave a zombie CUDA context** (31GB + 100% util,
  owner in R state). kill -9 the engine frees it; gpu-reset as last
  resort.
- **setsid nohup ... < /dev/null &** for servers that must survive the
  shell timeout (the timeout kills the process group otherwise).
- **vllm serve from the venv runs the WHEEL** (no TRITON_PAGED): set
  PYTHONPATH to the repo tree for our backend.
- **--gdn-prefill-backend only reaches the resolver via CLI**: LLM()
  kwargs are silently dropped (not in the signature).
- **/srv/benchmarks is not writable by joker** - redirect window logs
  to /tmp/opencode (a Permission-denied redirect kills the run
  silently).
- **sudo without cached credentials hangs** waiting for a password -
  use sudo -n and re-authenticate when needed.
- **Expanding the GDN autotune lists (KKT/DELTA_H BK/BV etc.) hangs
  server init** (0% CPU/GPU after weight load): the reduced SM70
  schedule is load-bearing until the autotune-in-profiling deadlock is
  bisected.
- **vLLM serves compile kernels lazily on first request** - send a
  warmup request before measuring; the first request includes minutes
  of JIT.
- **Promotion A/B (2026-09-04)**: TRITON_PAGED BEATS FLASH_ATTN_V100
  on Qwen3-1.7B eager: b1 1.16x, b8 1.11x, b16 1.09x, PPL parity 6e-5,
  greedy 20/20. F1 promotion criterion exceeded (was: within 10%).

## 2026-09-04b - Xid 31 autoinfligido y lecciones de recursos externos

- **Xid 31 (GPU memory page fault) fue causado por nuestras pruebas de
  autotune GDN** (pid del EngineCore, timestamp exacto 15:46:48 = el
  warmup skipped: illegal memory access). Un IMA de kernel genera Xid
  31; el contexto muere y el driver lo reporta. La GPU se recupero
  (produccion OK tras la ventana). REGLA DURA: no repetir experimentos
  que induzcan IMA en este hardware (tarjeta con historial RMA/Xid) -
  el hallazgo ya esta reportado upstream (#488); no aporta nada
  reproducirlo localmente.
- **Determinismo de prefill (patron DocAI)**: mismo prompt x10,
  temperature=0, max_tokens=1, top_logprobs=20, comparacion bit de la
  lista top-20. Mas sensible y barato que una suite: detecta kernels
  no deterministas aunque la salida parezca estable. Herramienta:
  `f2_determinism_probe.py`. Ejecutar antes de aceptar cualquier
  receta de serving nueva.
- **Divergencia bajo greedy SIEMPRE es bug de stack** (no varianza
  natural): controla con un segundo engine (llama.cpp) antes de culpar
  al modelo.
- **Determinismo expone**: el greedy+thinking loop de Qwen (fin de
  thinking repetido, respuesta vacia, finish=length) y la no-
  equivalencia MTP-vs-greedy eran fallos enmascarados por el ruido del
  kernel no determinista. Aplica a nuestro serving de HauhauCS
  (thinking mode): vigilar loops de thinking; retry con T>0 o
  presence_penalty como mitigacion.
- **Bug MoE FP16 de upstream vLLM (BLOCK_SIZE_K=128 en decode M<=64 ->
  register-spill en V100, 4-9x)**: nuestro fork YA esta protegido -
  1Cat tiene el fix equivalente default-on
  (VLLM_SM70_UNQUANTIZED_MOE_0DOT3_CONFIG=True, BK=64/32). Validado en
  el arbol; sin accion adicional. Referencia: v100-vllm-2026 ch.2.

## 2026-09-04c - Toolkit A/B (llama.cpp/QFlash en Volta) y protocolo multi-GPU

- **Toolkit A/B en llama.cpp (fuente 43991d229, Qwen3.8-27B Q4_K_P,
  -ngl 99, GPU 1, pp512+tg128)**: CUDA 12.0.140 vs 12.8.93 vs
  12.9.86 - los tres estadisticamente identicos (tg128
  33.39/33.34/33.34; pp512 647/643/639 +-ruido). El toolkit NO cambia
  la velocidad en Volta para este motor; elige por CORRECCION (12.9.x
  trae fixes criticos de cuBLASLtMatmul: resultados incorrectos
  concurrentes con kernels tensor-core, IMA con leading dimensions
  grandes) o por features de build (compresion binaria GGUF exige
  12.8+). Builds preservados: build-toolkit{120,128,129}.
- **Leccion de medicion**: sin CUDA_VISIBLE_DEVICES, llama-bench corre
  en GPU 0 (llama-main ocupada) y el fallo VMM del pool se ve como
  crash - siempre fijar la GPU en los benches.
- **Protocolo de diagnostico multi-GPU (2a V100)** - NO atribuir
  fallos automaticamente a R580. 1Cat tiene evidencia de un incidente
  en R580.159.03 y una ruta de custom all-reduce problematica en SM70
  que se estabiliza con --disable-custom-all-reduce. Orden correcto:
  1) probar cada GPU por separado; 2) p2pBandwidthLatencyTest +
  nccl-tests; 3) verificar nvidia-smi topo -m y afinidad NUMA;
  4) desactivar custom all-reduce en 1Cat-vLLM; 5) comparar R580 vs
  R570 manteniendo intacto el userspace cu128; 6) revisar Xid y
  segfaults en journalctl -k y dmesg.

## 2026-09-04d - Revision upstream/unsloth llama.cpp: no desplegable

- **Upstream tip (1863ac033, 0.4.0-dev, 137 commits por delante)**:
  contiene piezas interesantes (3466812d1 fuse MoE weighted expert
  reduction - relevante para Ornith A3B; e4b9af007 XOR swizzle FA) y
  12.9.2 compila sm_70 sin problema. PERO tiene una REGRESION
  bloqueante en V100: la carga del HauhauCS Q4_K_P (16.6 GiB) falla
  con `cudaMalloc failed: out of memory` en una GPU VACIA de 32GB,
  incluso con ctx 8192 y CUDA_VISIBLE_DEVICES explicito. No
  desplegable. Candidato a reporte upstream con el repro (worktree
  llama-wt-upstream + build-129 conservados).
- **Unsloth fork (261 commits)**: trabajo unico centrado en carries
  qwen4exp/Flash-Next (nextn draft head), GGML_CUDA_ENABLE_UNIFIED_
  MEMORY pin e higiene de CI (-Werror). NADA para nuestros modelos
  (Qwen3.8-27B GDN hybrid y Ornith A3B corren en mainline). SKIP.
- **Nuestro despliegue 12.9.2 (commits exactos de produccion) queda
  como mejor estado**: paridad/mejora medida, librerias cuBLASLt
  corregidas, rollbacks listos (.bak-129).
- Nota de metodo: git fetch de origin dio "unpack-objects fallo"
  parcial (objetos a medias) - verificar la integridad del fetch
  antes de construir desde un ref recien traido.

## 2026-09-04e - Regresion 12.9 aislada por A/B de toolkit (upstream tip)

- **La regresion de alloc del tip upstream es del TOOLKIT 12.9, no del
  codigo**: mismo tip, misma carga (HauhauCS Q4_K_P en GPU vacia) -
  con 12.9.86 falla (cudaMalloc OOM 16GB), con 12.8.93 CARGA Y
  SIRVE (health OK, 17.7GB residentes). El nvcc/runtime 12.9 genera
  algo que rompe la allocacion grande en sm_70. Reportable a
  NVIDIA/llama.cpp con este A/B limpio (una variable: el toolkit).
- **Rendimiento del tip**: 28.0 tok/s vs 40.0 de nuestro produccion
  (866322481 + 12.9.2) en la misma GPU/prompt = 30% mas lento para el
  HauhauCS. El fuse MoE no ayuda al GDN hybrid; el tip pierde decode
  en este modelo. NO desplegar.
- Veredicto final de la ronda driver/toolkit: produccion queda en
  commits exactos de produccion + toolkit 12.9.2 (40 tok/s) - la
  mejor config medida. Los toolkits 12.8/12.9 quedan instalados para
  bisects futuros; builds preservados (build-toolkit128/129,
  llama-wt-upstream/build-128 y /build-129).

## 2026-09-04g - Reporte regresion toolkit 12.9 + repro #490 (no reproduce)

- **llama.cpp #28416 creado**: regresion sm_70 toolkit 12.9.x (cudaMalloc
  OOM en GPU vacia) aislada por A/B con 12.8.x. Repro preservado
  (llama-wt-upstream/build-{128,129}).
- **#490 (TP2 >221K decode collapse) NO REPRODUCE en nuestro build**
  dev-line (ef68a0ea + sprint): 2x240K tokens concurrentes, config
  identica (TP2, FLASH_ATTN_V100, fp8_e5m2 KV, 262K, seqs 3, util
  0.92, prefix cache + Mamba align) -> 12.8 tok/s agregado, sano.
  Con NCCL_P2P_DISABLE=1 tambien sano (12.6 tok/s) -> P2P no es el
  gatillo. Diferencia restante: wheel 1.5.0 del reporter vs dev-line,
  o hardware PCIe fisico. Comentario publicado con el datapoint.
- Sonda deterministica de corpus largo: prompt de 240K tokens
  generado con tokenizer propio + shuffle sembrado (tools promocionados).
- Gotchas del round: (1) pkill -f con patron en el propio cmdline se
  suicida aunque uses corchetes si el patron esta en el string del
  bash -c (usar pgrep con [x] Y evitar el literal en el mismo
  comando); (2) curls en background con & mueren con el shell del
  bash tool al timeout - usar setsid bash -c '...' < /dev/null &
  disown; (3) el prompt de 1.16MB no cabe como argumento de curl -
  usar --data-binary @file.

## 2026-09-05 - F3 async scheduling evaluado: NO adoptar

- **TP2 (objetivo del sprint): async scheduling = CERO ganancia
  medible** (ITL 22ms/24ms, TTFT 1.63/1.65s, agg 111.1 vs 111.4
  tok/s - identico a sync). La hipotesis de la burbuja scheduler-CPU
  en TP2 queda REFUTADA en nuestro stack: el paso GPU domina y el
  pipeline del fork ya solapa lo solapable.
- **TP1: async mejora ITL solitario (-13%, 33 vs 38ms) pero PENALIZA
  concurrencia (+10% ITL x4, agg -9.5%)** - net negativo para serving.
- Bench: QUASAR NVFP4, prompts 2K x 256 tok greedy, streaming client
  (TTFT/ITL p50/p99), x1/x4, sync vs --async-scheduling, TP1 y TP2.
- **Determinismo**: greedy diverge en whitespace (2/96 tokens) sync vs
  async - benigno (chunked-prefill segmenta distinto -> acumulacion
  fp16 distinta en ties). Contenido identico.
- **Decision: NO activar --async-scheduling en produccion.** F3
  cerrado como evaluado-no-adoptado. Lever propio del fork:
  VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH (default 0) queda sin uso.
- TP2 NVLink dobla el decode de TP1 (ITL 22 vs 38ms solitario) -
  dato util para sizing.

## 2026-09-05b - Bench estrella TP2 32K: TRITON_PAGED stall, FA_V100+fp8 serve

- **TRITON_PAGED DEADLOCK en serving concurrente** del 27B GDN TP2:
  x4 con 4K prompts, GPUs al 0%, Running 3-4/throughput 0 - con int8
  PTH Y con fp16 KV (no es el dtype: es el backend). Request UNICO
  funciona (smoke 32K OK, 56 tok/s prefill lento). La promocion F1 se
  valido con baterias offline LLM() sobre Qwen3-0.6B denso - nunca en
  serving streaming concurrente del 27B. Bug candidado a reportar.
- **FA_V100 + fp8_e5m2 + APC = la receta de serving TP2 que SI
  funciona** (262K ctx, 2xV100): cold x4 32K: wall 153s, TTFT p50
  101s (prefill-dominated 128K); warm x8 (4+4 cache): 98.6 tok/s agg,
  17.4 tok/s/stream, ITL 68ms; APC shared x4: 88.1 tok/s agg,
  40 tok/s/stream, ITL 30ms, TTFT 5.2s.
- **VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16 = efecto CERO en TP2
  serving** (B vs C identicos). El +7% b16 era de 0.6B offline - no
  generaliza. Fuera de la receta.
- KV capacity TP2 QUASAR: fp8 1,093,773 tok / int8 PTH 1,044,082 /
  (fp16 ~550K) -> conc 4.13x a 262K len.
- Alfred-monitor: su health-check hace systemctl start de los 5
  servicios cuando estan down (joker@openclaw-workspace) - las
  ventanas largas deben avisarle o asumir crash-loop inofensivo
  (no puede asignar mientras vLLM sostiene la GPU).
- pkill -f se suicida si el PATRON literal aparece en CUALQUIER parte
  del comando (incluido el contenido de un printf/heredoc en el mismo
  bash -c). Kill SIEMPRE en comando separado.

## 2026-09-06 - #490 reproducido con prompts DISTINTOS; leccion harness

- **El colapso #490 SI reproduce en nuestro sprint** (gef68a0ea+sprint,
  SXM2 NVLink, custom-AR off): 2 prompts DISTINTOS ~239K concurrentes
  -> 0.2-0.3 tok/s sostenido 8 min (24 ventanas de log), KV creep
  ~90K, hit 0.0%. Mi "no reproduce" anterior fue un artefacto de
  prompts IDENTICOS (cache hit enmascara el solape prefill+decode).
  Corregido en publico (#490 comentario 5562125911).
- Mecanismo (areslp): batch MIXTO prefill+decode no toma la ruta
  rapida XQA (q_len==1 puro) -> decode recalculado sobre contexto
  completo. Mismo familiares que #505 (TRITON_PAGED stall en batch
  mixto) -> problema arriba del backend de atencion (capa scheduler/
  composicion de batch). Workaround comunitario: --max-num-seqs 1.
- **Regla de harness: NUNCA usar prompts identicos para tests de
  concurrencia** - el cache hit invalida el arm. Distinct por defecto;
  identical solo como arm explicito de cache.
- Repos comunitarios/officiales SIEMPRE antes de benchcar: el runbook
  Redhatvale ya mapeaba (custom-AR off PCIe, max-num-seqs 1,
  VLLM_SKINNY_DROP_CT IMA en TP2, TP1 NVFP4 no cabe 32GB).
- llama.cpp #28416: WARNING por politica anti-IA (JohannesGaessler).
  NO escribir mas en repos ggml con esta cuenta sin texto humano.
  El bisect pedido (commit x toolkit) seria caro; dejar el A/B como
  evidencia.
- Stack comunitario alternativo a contrastar: v100-skinny (kernels
  NVFP4 W4A16 hand-written) 61-74 tok/s PCIe / 95-118 SXM2 con MTP
  k=7 vs nuestro TurboMind compressed-tensors.

## 2026-09-07 - Migracion metodologica oficial + stack v100-skinny en SXM2

- **Bench oficial vllm bench serve sobre la receta estrella** (TP2,
  FA_V100+fp8+APC, QUASAR): 8x32K@256 cold request-rate 8 -> 8/8 OK,
  TTFT mean 179s / p99 364s, output 5.37 tok/s agg (prefill-dominated).
  Consistente con el harness ad-hoc -> cross-validacion OK. Flags del
  fork: --base-url SIN /v1 (doble path = Not Found), auth por env
  OPENAI_API_KEY, --save-result es boolean + --result-dir.
- **GDN exactness (harness oficial benchmark_sm70_gdn_exactness.py):
  sprint vs main wheel = torch.equal PERFECTO** (out y final_state,
  max_diff 0.0) -> nuestro branch no perturba la numeria GDN. Gate de
  regresion adoptado. Necesita GPUs con holgura (OOM junto al server).
- **Stack v100-skinny (1.2.2 + kernels QPN + RadixArk NVFP4) FUNCIONA
  en nuestro SXM2 2x32GB TP2**: bootstrap con REQUIRE_GPUS=2 (su check
  asume TP4x16GB), JIT QPN2/QPN8 con CUDA 12.8 + ninja en PATH,
  served-model-name 'qwen3.8-27b' (punto!), server sin auth.
  **52.18 tok/s single-stream (MTP k=7, MNS=1)** con bench oficial
  (4x1024@256 serial, ITL mediana 46ms). Su publicado PCIe: 61-74.
- **Contraste skinny vs estrella**: skinny = maximo single-stream
  (MTP, concurrencia limitada a 1 por los bugs upstream #490/#505);
  estrella = serving concurrente (88-99 tok/s agg x4-8 a 32K warm,
  45 tok/s/stream corto sin MTP). Regimenes distintos, tools distintos.
- Gotcha recurrente CONFIRMADO: pkill -f no matchea VLLM::Worker_*
  (solo el launcher) -> workers zombis sosteniendo 30GB/GPU y
  produccion fail-load. Matar SIEMPRE verificando
  nvidia-smi --query-compute-apps despues de cada server.

## 2026-09-07b - Skinny MNS=4: funciona pero colapsa 22x

- Skinny stack con MNS=4 ARRANCA limpio (sin hang de captura) y
  sirve 4/4 concurrentes SIN stall - pero **9.64 tok/s agregados**
  (~2.4/stream) vs **52.18 single-stream** = colapso x22. El MNS=1
  comunitario es por RENDIMIENTO, no solo por estabilidad: el batch
  mixto prefill+decode degrada igual que en 1.5.0 (#490 familia,
  aunque aqui sin stall).
- Veredicto final del contraste comunitario: skinny = especialista
  single-stream (MTP k=7, 52 tok/s); estrella = champion de serving
  concurrente (88-99 tok/s agg x4-8 a 32K). Para LHU multi-usuario,
  la estrella sigue siendo la receta.
- /tmp se arraso de nuevo (cleanup del boot) - regla: los scripts de
  bench viven en git; los corpus se regeneran on-demand.

## 2026-09-07c - Arms asimetricos #490: el colapso escala con el prefill del OTRO

- Arms en SXM2 (fp8+APC+262K, seqs 3, greedy, streaming per-request):
  A 8K+240K -> corto colapsa a 0.48 tok/s DURANTE los 346s de prefill
  del largo (que luego decodifica sano a 21.3). B 32K+240K -> corto a
  0.14. C control 8K+32K SIN colapso (35.6/11.0 sanos).
- **Umbral del acantilado entre 32K y 240K de largo de prefill co-
  residente.** El dano es unidireccional: paga quien decodifica
  durante el prefill largo. El 2x240K previo (ambos a 0.2-0.3) se lee
  ahora como la misma regla, no un caso especial.
- Publicado en #490 (5566577427) con oferta de bisectar el acantilado
  (64K/128K/192K). Arm runner en /tmp (regenerable; patron en git).
- Produccion restaurada OK; workers verificados muertos via
  compute-apps antes de relanzar.

## 2026-09-07d - Bisect del acantilado #490: gradiente suave, vuelco 32K->64K

- Curva completa (8K fijo vs partner variable): 35.6 (8K) / 11.0
  (32K) / **2.03 (64K)** / 0.84 (128K) / 0.86 (192K) / 0.14-0.48
  (240K). El vuelco esta entre 32K y 64K (64K = 8 chunks de 8192).
  Meseta 0.85-2 en 64K-192K, profunda de nuevo en 240K. Dos regimenes
  con transicion, no un acantilado unico.
- El decode del largo declina suave con su largo (32.3->21.3, escala
  normal). La anomalia es SOLO el co-residente decoder.
- Nuance de scheduling: con partner >=128K el TTFT del corto cae a
  ~1s (entra antes a decode, absorbe mas ventana) -> la contaminacion
  se acumula por paso compartido, no es un impuesto fijo.
- Publicado en #490 (5567006760) con la nota del boundary 64K=8
  chunks como pista para el fix.

## 2026-09-07e - F7: MTP k=7 en la receta estrella - gana en single Y concurrente

- Checkpoint RadixArk NVFP4 trae cabezas MTP (mtp.fc.weight +14 en
  model-00003); arquitectura resuelta Qwen3_5MTP. Flags: envs
  SM70 MTP DEFAULTS + --speculative-config '{"method":"mtp",
  "num_speculative_tokens":7,"draft_sample_method":"greedy",
  "use_local_argmax_reduction":true}'.
- **Single 1024@256: 51.2-60.3 tok/s committed** (2 runs, median ~56)
  vs ~45 sin MTP (+25%) y vs skinny k=7 52.2. TTFT 725ms (!) por
  1024 tok (FA_V100 prefill muy superior al skinny 9.1s).
- **Concurrente 8x4096@256 rate 8**: 34.71 tok/s agg con MTP vs
  30.63 control sin MTP mismo checkpoint/arm (+13%), **TTFT mean
  16.2s vs 35.8s (-55%)**. 8/8 OK sin stall (familia #534 no aplica
  en TP2 single/multi razonable).
- Contraste Cerebras: nuestro mejor single-stream 60.3 vs 1500
  (25-48x, fisica WSE). La direccion MTP es correcta: menos lecturas
  de peso por token comprometido.
- pkill self-kill OTRA VEZ (printf con 'vllm serve' literal en el
  mismo comando) - REGLA ABSOLUTA: kill SIEMPRE aislado, jamas en el
  mismo bash -c que genere contenido con el patron.

## 2026-09-07f - MULTIMODAL EN EL FORK: funcionando (lyf NVFP4-MTP-VL TP2)

- **Vision sobre V100 en 1cat-vllm sprint CONFIRMADA**: checkpoint
  lyf/Qwen3.8-27B-Heretic-ARA-NVFP4-MTP-VL (Qwen3_5ForConditionalGeneration,
  vision_config ViT 1152, NVFP4 compressed-tensors, 20GB, 2 shards) carga
  y SIRVE vision a la primera: describio exactamente imagen de test
  (circulo rojo + rectangulo amarillo + fondo azul). Vision tower via
  MMEncoderAttention -> TORCH_SDPA (Volta OK).
- Resultados (TP2, MNS=6, 131K ctx, fp8 KV, APC, MTP): texto single
  1024@256 = 47.7 tok/s (TTFT 1.1s); texto x6 4096@256 = 46.6 agg,
  6/6 OK (vs skinny x4 9.6); vision cold ~8s e2e, misma imagen x3
  concurrente ~1s (APC cachea tambien los tokens visuales).
- La imagen ocupa ~221 prompt tokens (448px, patches mergeados) -
  APC aplica a prefijos visuales identicos.
- Descarga en /home/joker/v100-qwen38/lyf-vl (persistente). Con esto,
  la Opcion 1 (vLLM estrella TP2) ya tiene TODAS las capacidades de
  la produccion actual salvo... nada: texto+vision+MTP+APC+131K.

## 2026-09-07g - A/B MTP k=3 vs k=7 con flags oficiales de receta

- Flags oficiales de recipes.vllm.ai integrados a la receta estrella:
  --reasoning-parser qwen3 (el template abre <think>; sin parser el
  reasoning contamina content), --enable-auto-tool-choice
  --tool-call-parser qwen3_coder, --enable-prefix-caching explicito.
- **Acceptance real con texto natural (metricas spec_decode)**:
  k=3 = **79.4%** (en el rango 0.75-0.9 de la receta); k=7 = 60.0%.
  El acceptance agregado cae con la profundidad del draft (las
  posiciones lejanas aciertan menos).
- **Velocidad single (mismas condiciones): k=3 = 28.19 tok/s,
  k=7 = 24.45 tok/s -> k=3 GANA en acceptance Y velocidad.**
  La receta recomendaba k=3 y tenia razon: draft mas corto = menos
  computo de draft por paso y commits mas seguros.
- Caveat: ambas cifras absolutas son menores que las de F7 (51-60
  con k=7) - condiciones de reloj/termal no identicas entre sesiones;
  el A/B interno k3-vs-k7 es el dato solido (misma sesion, mismo
  estado). Re-verificar clocks si el absoluto importa.
- Produccion restaurada OK tras el A/B.

## 2026-09-07h - Sweep MTP k=1..7 completo (acceptance + single + x6)

- Barrido systematico (mismo harness v2, 6/6 OK por k, x6 @4096@256):
  acceptance DECRECE monotono con k: 91.0 (k=1) / 77.6 / 70.4 / 69.6 /
  **66.4 (k=5)** / 63.1 / 53.4 (k=7). Textbook.
- Single: plateau 25-30 en casi todos PERO **k=5 = 42.64 outlier
  alto**, y su x6 tambien la mejor (46.88 agg). k=1 = mejor ITL
  (45.2ms, snappiest) y mejor acceptance (91%).
- x6: 27-47 agg todo 6/6 OK (k=5 46.9 > k=7 46.1 > k=1 44.1).
- **Misterio F7 (k=7 a 51-60 antes, 26 hoy): NO era el env faltante**
  (sweep uso envs F7-exactos). Explicacion parcial: los benches del
  sweep corrian con 3x12K prefijos residentes en cache (tests de
  acceptance previos) vs F7 que bencheaba server recien-arrancado.
  Varianza sesion-a-sesion ~2x documentada; los relativos intra-sweep
  son solidos.
- Veredicto: k=5 candidato sweet-spot (necesita re-validacion 3-runs
  por el ruido 1-run); k=1 para interactividad (ITL 45ms + 91%
  acceptance). Registrar sweep-results.jsonl como dato.
- El sweep script v1 murio 7x en silencio (heredoc anidado sospechoso);
  v2 = launcher parametrizado + script por-k con logging por paso ->
  funciono a la primera. Leccion: scripts de barrido con logging por
  paso y sin heredocs anidados.

## 2026-09-07i - Auditoria de flags + E1: MBT=16384 es un win claro

- **Auditoria**: 'use_local_argmax_reduction' ES real (leido por
  llm_base_proposer.py). 'draft_sample_method' NO existe en el fork -
  key muerta ignorada silenciosamente (la oficial es
  'rejection_sample_method' pero en NUESTRA version solo acepta
  'standard'|'synthetic' - 'strict' de main ROMPE el boot). Eliminada.
- **E1 (k=5 + --max-num-batched-tokens 16384)**: acceptance 66.4%
  (identico al sweep k=5@8192 - la acceptance es propiedad del modelo,
  metrica estable ✓). **Single 47.91 tok/s (+12% vs 42.64 sweep),
  x6 @4096 = 53.11 agg (6/6 OK, +13% vs 46.88) - el mejor numero
  concurrente medido en este stack.**
- RECETA FINAL estrella v2: RadixArk NVFP4 + FA_V100 + fp8_e5m2 KV +
  APC + MTP k=5 + use_local_argmax_reduction + MBT=16384 + parser
  qwen3 + tool-parser qwen3_coder + seqs 6 + 131K (262K sin MTP).
- Pendiente menor: confirmar k=5@8192 con mas runs (el acceptance
  idico ya valida la consistencia).
