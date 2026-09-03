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
