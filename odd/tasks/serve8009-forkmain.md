# Feature: serve8009-forkmain

Objective: upgrade the tree serving :8009 (llama-vllm-star.service) to fork/main
with the chunked-cast drafter patch, probe custom-AR-SM70 in the window, and
adopt ctx 262k only with proven VRAM margin. Hard requirement (owner): no more
drafter OOMs; every decision backed by measured data.
Branch: sprint/f1-triton-attn -> prod/forkmain-20261027 (in
/srv/benchmarks/lhu/upstream/1cat-vllm, checkout of SabaTech-dev/1Cat-vLLM)
Delivery: single restart window tonight; A/B legs; docs + Engram at close.
Test runner: functional canary + A/B bench (single decode t/s, 4-concurrent
burst, VRAM peaks via nvidia-smi polling); no unit framework applies (ops).

## Specs

S1: User (verbatim): "vamos a poner al día nuestro fork de 1cat-vllm y mejorar
    lo que servimos en el puerto :8009".
S2: User choice (interactive round): target base = fork/main (198919b21);
    improvements = probe custom-AR + "Tienes que optimizar y con datos reales,
    no queremos mas OOMs" + "Probar ctx 262k"; timing = "Ahora".
S3: Serving chain (verified): unit llama-vllm-star.service sets
    PYTHONPATH=/srv/benchmarks/lhu/upstream/1cat-vllm:.../flash-attention-v100
    -> venv /srv/benchmarks/1cat/venv -> editable finder (build-1cat dev381) is
    SHADOWED by PYTHONPATH; live workers import
    /srv/benchmarks/lhu/upstream/1cat-vllm/vllm (journal traceback 27-oct
    14:14/19:38, line 959 qwen3_dflash.py; /proc/PID/maps confirms).
S4: Patch to carry: chunked FC projection in combine_hidden_states
    (qwen3_dflash.py, +25/−6, bitwise-verified on CPU by the night session,
    peak 205 MB -> 84 MB per chunk). fork/main's copy is NOT chunked -> port
    required (function moved to line ~987, context changed +80/−30 vs
    2980cc931).
S5: csrc delta 2980cc931 -> main = 46 files +25,540/−400 -> rebuild of
    compiled ops is MANDATORY before restart (new torch_bindings
    registrations vs old .so = missing-symbol crash risk).
S6: Rejection with data: e5m2-XQA-decode NOT applicable (target model
    coder390 NVFP4: text_config 24 heads / 4 KV = q_per_kv 6; community patch
    is for q_per_kv==4). Sampler-sync #1267 NOT applicable (fork/main has no
    compact top-k/top-p `.all()` path).
S7: Acceptance gate for ctx 262144: OOM-shape test (4 concurrent ~36k-token
    prefills + decode) must pass with post-warmup VRAM headroom >= 1.5 GiB on
    both GPUs and zero OOM events; otherwise the hardening default (200k)
    stays.
S8: custom-AR adoption gate: probe can_device_access_peer(0,1)==True AND
    A/B shows @4c agg >= baseline (no regression); VLLM_SM70_TOP1_CUSTOM_AR=1
    goes into a separate drop-in leg first, default stays off unless it wins.
S9: Rollback at any point: git checkout sprint/f1-triton-attn in the serve
    tree (patch branch preserved), restore drop-ins, restart. Hardening
    drop-in zz-zz-hardening-20261027.conf is the fallback config.

## Tasks

T1 [S3] Baseline bench of running serve (single + 4c burst + VRAM peaks) — inline
T2 [S2,S8] Stop serve + probe can_device_access_peer(0,1) — inline
T3 [S1,S4] Commit patch branch (preserve), checkout fork/main, port chunked-cast, push — inline
T4 [S5] Rebuild compiled ops in venv, verify import paths — inline (long)
T5 [S2] Restart with hardening 200k + canary (health, greedy, T>0) — inline
T6 [S2,S7,S8] A/B legs: new-base vs baseline; custom-AR leg if probe True; 262k OOM-shape leg — inline
T7 [S1] Final config decision + docs (ROADMAP addendum, daily, Engram) + report — inline

## Log

L1 (2026-10-27): User request verbatim: "Usa el sitema de preguntas
    interactivas con opciones detalladas y tu recomendacion y vamos a poner al
    día nuestro fork de 1cat-vllm y mejorar lo que servimos en el puerto
    :8009". Answers: fork/main (Recomendada); mejoras: Probe custom-AR
    (Recomendada) + "Tienes que optimizar y con datos reales, no queremos mas
    OOMs" + "Probar ctx 262k"; ventana: "Ahora (Recomendada)".
L2 (2026-10-27): Fact-check resolved the serving chain (S3): PYTHONPATH in the
    unit shadows the dev381 editable; live code = lhu tree + dirty patch
    (LIVE since ~22:0x restart). Sweep conclusions reused: fork/main has
    #561 (drafter memory, root-cause family) and NOT the Oct regressions
    (#1269 corrupto DFlash2 T>0, #1042, #1220).
L3 (2026-10-27 23:0x): Baseline (patch live, 200k): single 176.03 t/s,
    4c agg 107.76 t/s, TPOT 4c 9.32 ms, VRAM estable 31443/32199 MiB.
    Probe custom-AR: can_device_access_peer(0,1)=True (both ways) — REFUTA la
    hipótesis del sweep 19-oct (False esperado por VT-d); leg custom-AR GO.
    Migración: rama parche sprint/f1-dflash2-chunked-cast 30dd61ae1 + prod
    branch prod/forkmain-20261027 f1e8e9920 (port chunked-cast, diff CPU
    3.7e-08), ambas pusheadas. Rebuild: 1er intento falló (gcc-15 vs nvcc
    12.9, Ubuntu 26.04); relanzado con CC/CXX/CUDAHOSTCXX=gcc-13 (Engram
    5136). .so respaldados en /tmp/opencode/so-backup.
L4 (2026-10-28): Los 2 reboots del 27 (23:33/23:52, sin causa en journal, ambos
    bajo build -j32) dejaron: JIT de flash_qla (GDN sm70) con lock huérfano
    en ~/.cache/torch_extensions (lock del 19-sep que solo bloqueaba al entrar
    en rama de recompile — hash nuevo con fork/main) + gcc-15 del JIT fallaría
    (nvcc 12.9). Fix: lock eliminado + drop-in zz-z-jit-toolchain.conf
    (CC/CXX/CUDAHOSTCXX=gcc-13). Serve restaurado 16:32: health 200, canario
    verde (greedy OK; T=0.7 con DFlash2 57,9 t/s — el port chunked-cast
    funciona). 16h de servicio caído (23:52->16:32). Build suave relanzado
    (MAX_JOBS=8, log rebuild-forkmain-20261028.log).
L5 (2026-10-28 19:0x-19:3x): Fix CMake d3d86c9ae (activation_kernels en sm70)
  → _C carga OK, serve 19:08. PERO canario T=0.7 CORRUPTO (firma #1269:
  secuencias ASCII, greedy limpio) y single 108 t/s (−39%). Diferencial
  argmax: LIMPIO a T=0.7 y T=1.0, y recupera baseline (173,08 single /
  105,94 conc4 / 9,48 ms). ⇒ El bug de corrupción del draft sampler
  probabilístico SÍ está en fork/main (vía distinta de c4f6245f84 — el
  chequeo de ancestro era insuficiente). Hipótesis de causa raíz: con el
  .so viejo el op torch.ops._C.silu_and_mul_quant no existía y el pase de
  fusión act_quant_fusion.py saltaba esas fusiones; con el .so nuevo el
  pase fusa el kernel (¿roto en Volta?) en los grafos del drafter →
  argmax enmascara, sampling expone. Confirmación pendiente (micro-test
  del kernel con GPUs libres, 5 min). Config de prod mientras tanto:
  drop-in zz-zz-argmax-test.conf (draft_sample_method argmax) — medida y
  estable. Pierna A (argmax): 173,08/105,94/9,48. Build de campaña
  upstream/main 4d57473db6 lanzado (prod/upstream-main-20261028
  e4f312e337, con port chunked-cast; upstream ya arregló el bug de CMake);
  su gate incluirá el MISMO test T>0.
L6 (2026-10-28 21:4x): CAUSA RAIZ confirmada por micro-test GPU: el op
    silu_and_mul_quant compilado en sm70 calcula ~1/3 del valor en ~45% de
    elementos (maxdiff 24.0; repro en Engram 5141). El greedy sampler NO
    arregla nada (T=1.0 degenerado tambien con greedy: 3/4 muestras mal —
    el drop-in "argmax" previo nunca se aplico por orden lexico, los boots
    de 19:10/20:51 eran probabilistic: comparacion invalida). Campana
    upstream: NO-GO confirmado (corrupcion T=1.0 con su CMake que compila
    el fichero roto incondicionalmente en sm70). Fix correcto fork/main
    (pendiente, diurno): NO registrar los ops de activation_kernels en
    torch_bindings en el else-branch sm70 (la fusion salta si el op no
    existe). Rollback en marcha: rebuild de sprint/f1-dflash2-chunked-cast
    (arbol probado + parche OOM) con la receta; al terminar restart +
    canario multi-muestra T=1.0 + bench = prod restaurada.
L7 (2026-10-28 22:4x): CIERRE. Prod restaurada y validada: serve tree en
    sprint/f1-dflash2-chunked-cast @ 30dd61ae1 (csrc recompilado con la
    receta gcc-13/arch7.0), drop-ins de prueba todos .disabled, ExecStart
    efectivo = hardening 200k (probabilistic default, como el prod estable
    pre-upgrade). Validacion final: 6/6 muestras T>0 coherentes (sin basura
    ASCII/loops), bench 164,94 single / 103,36 conc4 / 9,71 ms (≈baseline
    176,03/107,76/9,32; delta −4/−6% atribuible a toolchain de rebuild).
    Veredictos: fork/main = NO-GO hasta parche no-register (diurno);
    campaña upstream/main = NO-GO (hereda kernel roto + #1269); sprint+patch
    = GO. Trabajo preservado en origin: sprint/f1-dflash2-chunked-cast
    30dd61ae1 y prod/forkmain-20261027 f1e8e9920(+d3d86c9ae). Pendientes
    diurnas: (1) parche no-register de activation_kernels ops en sm70 para
    rescatar fork/main; (2) reporte upstream #1269 con repro del kernel
    roto (prouesta al dueño); (3) A/B del efecto toolchain en single (−6%).
L8 (2026-10-29 02:0x-03:4x): RESCATE de fork/main en dos parches mas:
    (1) act_quant_fusion.py accedia EAGER al op ausente en importacion =>
        AttributeError en worker init; fix con el patron hasattr del propio
        upstream (8932de81d) — la entrada kFp8StaticTensorSym y su registro
        de patron ya estan key-gated (lineas 63 y 96 del fichero).
    (2) fork/main rescatado = COHERENTE 6/6 a 0.6/0.7 pero LENTO (83-85
        single, -53%; conc4 -7%) — la fusion saltada u otra regresion de la
        sync 17-sep; investigacion diurna.
    Decision de prod: swap final al arbol sprint (rebuild + restart 03:03):
    6/6 coherente; bench inicial 27,9 = throttling termico por caja
    caldeada (GPU0 83C, HwThermalSlowdown activo, reloj 1335/1530; tras
    enfriar: relojes 1530 y flags 0x0). Remeasure de validacion final en
    curso tras 20 min de enfriamiento. Config final de prod: sprint
    30dd61ae1 + hardening 200k + genconf-cap 0.6 + LiteLLM 0.6 + OpenCode
    200k. Parches de fork/main preservados en origin (ebeb2614e +
    8932de81d) para el resgate definitivo cuando se entienda el coste -53%.
L9 (2026-10-29 09:3x): RESOLUCION del "-53%": NUNCA fue fork/main ni el
    kernel — el drop-in zz-zzzz-genconf-cap.conf redefinia ExecStart copiado
    a mano y OMITIA --speculative-config => el serve corria SIN DFlash2
    desde 23:41 del 28-oct (single no-spec ~83-86 = exactamente la mitad;
    conc4 casi no cambia, consistente con "MTP ayuda single"). Regenerado
    programaticamente desde el hardening canonico (sed de UN solo cambio) +
    diff token-a-token antes de aplicar: **177,51 / 107,85 / 9,31 ms =
    baseline restaurado al 100%** en sprint. Leccion (Engram 5148): jamas
    copiar ExecStarts a mano; generar con sed de un cambio + diff.
    Fase 2 (reporte upstream): draft en
    /srv/qflash/upstream-comment-draft-1269-20261029.md — pendiente
    aprobacion de Joker para publicar. Fase 4: drop-in 262k regenerado
    (disabled, sin draft_sample_method). Fase 1 en curso: rebuild de
    fork/main rescatado; al terminar: canario 0.6/0.7 + bench — si ~172 y
    coherente => swap de prod a fork/main. Fase 3 (custom-AR) sobre el
    ganador.
L10 (2026-10-29 12:4x): FASES 3-4 cerradas.
    Fase 3 custom-AR: EMPATE TECNICO (176,71/107,22/9,36 vs 175,32/109,01/
    9,21 — todo ruido) => NO se adopta (default off, documentado). En este
    workload el allreduce no es cuello.
    Fase 4 262k: el test de forma OOM (4x36k, 114k tokens) PASO sin OOM
    (3.510 tok/s prefill) PERO margen post-test GPU1 = 391 MiB (< 1536 del
    criterio S7) => NO se adopta; se mantiene 200k (hardening probado).
    PROD FINAL: fork/main prod/forkmain-20261027 @ 8932de81d (+ act_quant
    fusion fix) + hardening 200k + DFlash2 + genconf-cap 0.6 + LiteLLM 0.6
    + OpenCode 200k. Validado: 175,32/109,01/9,21 warm, 6/6 coherente
    0.6/0.7, health 200. UPGRADE ATERRIZADO: #561 + reduccion LM-head +
    port chunked-cast + fixes corrupcion, todo en prod.
L11 (2026-10-29 19:4x): REGRESION de memoria en fork/main detectada en
    produccion: 98 OutOfMemoryError hoy desde el swap (12:32), fatal error
    14:05, crash 19:06 (Worker_TP1: pide 200 MiB con 179 libres en GPU1,
    la de embeddings/drafter) bajo el bench T-Bench n=4, health 000
    (crash-loop). El sprint en la misma config (200k/4096/seqs4/0.90)
    aguanto toda la tarde sin un solo OOM. Veredicto: fork/main tiene
    menos headroom efectivo (paginas compact-hybrid / contabilidad de pool
    distintas) => ROLLBACK de prod a sprint (rebuild en curso, -j16) y
    fork/main queda como I+D con sus 2 parches de corrupcion preservados.
    Pista diurna: bajar util (0.90 -> 0.87-0.88) o investigar el sizing
    del pool de fork/main; el watchdog del bench funciono perfecto
    (aborto harbor al morir el motor — proteccion diseniada y probada).
    El bench T-Bench queda APARCADO hasta que fork/main sea seguro o se
    defina estrategia (sprint + bench, con los riesgos conocidos del
    drafter ya mitigados por chunked-cast).
L12 (2026-10-29 20:2x): CIERRE FINAL. Prod restaurada en sprint
    30dd61ae1 y validada en caja fria: 176,02 single / 110,47 conc4 /
    9,09 ms TPOT (>= baseline), 4/4 coherente 0.6/0.7, GPUs 59/58C.
    Resumen del arco completo: (1) upgrade intentado -> 4 escollos de
    build resueltos y documentados; (2) causa raiz del bug DFlash2 T>0
    reproducida a nivel kernel (silu_and_mul_quant roto en sm70) y
    REPORTADA upstream (#1269 comentario 6494824549, firmado SabaTech-dev)
    + PR #1322 (draft) con el fix de build; (3) fork/main rescatado
    funcionalmente (2 parches) pero con REGRESION de memoria bajo carga
    (98 OOMs vs 0 en sprint => rollback); (4) T-Bench aparcado hasta
    fork/main seguro; (5) estudio 2xTP1 y Flash-Next commiteado
    (ESTUDIO-2xTP1-vs-FlashNext-20261029.md). Pendientes diurnos:
    investigar la regresion de memoria de fork/main (util 0.87-0.88 o
    sizing del pool), vigilar #1269/#1322, ventilacion de la caja.
L13 (2026-10-29 20:4x): A/B sin drafter (peticion Joker "ver si sube la
    conc4 agregado"): sprint SIN DFlash2 = single 78,13-83,56 / conc4
    83,87-94,99 / TPOT 10,57-11,97 => la conc4 BAJA -12/-17% y el single
    -53%. DFlash2 GANA tambien en concurrente a ctx corto/medio (el patron
    "MTP perjudica concurrencia" de upstream NO se da en nuestro workload
    actual). Drafter restaurado y confirmado (172,45/108,49/9,25,
    speculative-config activo). Drop-in no-spec archivado como
    zz-zzzzz-nospec-test.conf.disabled (reutilizable para futuros tests).
