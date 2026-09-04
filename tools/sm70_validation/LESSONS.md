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
