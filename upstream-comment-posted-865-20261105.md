# PUBLICADO el 2026-11-05 20:47 UTC: https://github.com/1CatAI/1Cat-vLLM/issues/865#issuecomment-6668997727 — comentario original para 1CatAI/1Cat-vLLM#865 (GO del dueño pendiente — acción externa)

> Posteador: SabaTech-dev (como el comentario de #1269). Al pegar, eliminar este
> encabezado. Fecha del borrador: 2026-11-05. Datos en English (idioma del issue).

---

Same crash class also reproduces on a different SM70 topology — dense TP2, not
Flash-Next TP4:

- 2x Tesla V100-PCIE-32GB, TP2, PCIe (no NVLink), power caps 200 W
- 1Cat-vLLM fork build `v1.5.1.dev91+g30dd61ae1` (DFlash2 drafter), torch 2.10.0+cu128
- Model: dense coder390 NVFP4 W4A16 (compressed-tensors) + DFlash2 drafter
- 200k ctx, `--kv-cache-dtype fp8_e5m2`, `max-num-seqs 4`, `--enable-prefix-caching`

**Symptom**: engine dies with `TimeoutError` in
`multiproc_executor.get_response` → `shm_broadcast.dequeue` while waiting for the
TP workers: workers stop draining steps (EngineCore logs *"No available shared
memory broadcast block found in 60 seconds"*, then the hard deadline raises).
APIServer's `AsyncLLM output_handler` fails right after. **8 occurrences in 48 h**
on a production box serving live traffic, all with **1-2 running requests and
`scheduled_new_reqs=[]`** in the scheduler dump — i.e. ordinary decode steps, no
new input (one occurrence did include a 15.3k-token image prompt, but mm is not
required to reproduce). No Xid, no host OOM, workers log **no traceback** before
stalling — they go silent.

Recovery is watchdog-driven: systemd `Restart=always` does not act in every case
(the APIServer process sometimes hangs instead of exiting after EngineCore death,
so the port stays closed and the unit stays "active").

Two questions:
1. Is the DFlash/GDN decode path on SM70 known to stall under ≥2 concurrent
   requests in **TP2 dense** setups too, or is the Flash-Next TP4 case the only
   confirmed repro so far?
2. Any known mitigation short of disabling the drafter (single-stream works fine
   here as well, but that halves our single-stream throughput)?

Full py-spy dumps + scheduler dumps from the stalls are available on request.

---

## ADENDA 05-nov noche — RCA capturado EN el stall (py-spy, stress conc4 tg512 en vivo)

Reproduje el stall en 40 min con 4 concurrentes × 512 tokens sostenidos (131
rondas): throughput sano ~147 t/s → soak a 82 °C de GPU → **generation counter
congelado con /health devolviendo 200** (el APIServer sigue vivo) → todas las
peticiones fallan. py-spy capturado EN el stall (los workers siguen vivos):

- **Worker_TP0** (active, MainThread):
  `compute_common_gdn_attn_metadata` (`vllm/v1/worker/gpu/attn_utils.py:488`)
  ← `prepare_attn` (`mamba_hybrid.py:371`) ← `execute_model` — atascado en la
  construcción de metadatos GDN con `torch.cumsum` sobre `spec_sequence_masks`
  (máscaras de spec-decode).
- **Worker_TP1** (active+gil): dentro del **replay de CUDA graph de inductor**
  (`static_cuda_launcher.py:269` → `piecewise_backend.py:380` → `cuda_graph.py:303`).

Los dos ranks están **divergidos en fases distintas del pipeline** (TP0 en
Python computando metadatos de la siguiente pieza; TP1 lanzando un graph) —
deadlock, no lentitud. `EngineCore` muere por el timeout de `execute_model`
(`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`) y el APIServer cuelga sin salir.
**No térmico**: en el momento del stall las GPUs estaban a 49/42 °C a máximos
relojes en idle-spin (hubo soak previo de 30 min a 82 °C, sin throttling activo).

Dumps completos de py-spy + scheduler dumps disponibles bajo petición.
