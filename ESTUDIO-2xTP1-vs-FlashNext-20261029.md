# Estudio: alternativas de servido del modelo en :8009 — 2×TP1 vs Flash-Next (29-oct-2026)

> Pregunta del dueño: (A) servir el modelo actual uno en cada GPU (2 instancias
> TP1) — ¿concurrency/velocidad/contexto? · (B) usar Qwen3.8-Flash-Next
> (Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata).
> Rig: 2× V100-32GB PCIe (sin NVLink activo), TP2 actual.

---

## A) Dos instancias independientes (1 modelo por GPU, TP=1)

**Memoria por GPU (TP1)**: pesos completos ≈ 14,5 GiB + drafter DFlash2
completo ≈ 2-3 GiB + activaciones/graphs ≈ 1,5-2 GiB → KV pool restante
≈ 12-13 GiB. A fp8_e5m2 el modelo gasta ~23,3 B/token (4 cabeceras KV
completas en 1 GPU vs 2 en TP2) → pool ≈ **530-570k tokens por instancia**
(comparable al pool total del TP2 actual: 547-608k, porque en TP2 la otra
GPU gasta su memoria en la otra mitad de pesos/drafter).

**Velocidad por petición**: el decode es bandwidth-bound → TP1 lee TODOS los
pesos por paso de una GPU (900 GB/s) vs TP2 lee la mitad de cada una
(1,8 TB/s agregados) → **TP1 ≈ la mitad de velocidad por request**:
- TP1 + DFlash2 single: **~85-90 t/s** (vs 175 TP2)
- TP1 sin spec: ~41-43 t/s

**Concurrencia**: 2 instancias × 4 seqs = **8 secuencias totales** (vs 4).
Pero el agregado NO mejora: mismo ancho de banda y cómputo total →
agregado ≈ 100-110 t/s, **igual que el TP2 actual (109)**. Confirmado por
datos propios: el no-spec TP2 (83-86) ya demostró que sin drafter el TP2
agregado no cae.

**Contexto**: 200k por petición sigue siendo viable por instancia (pool
~550k tokens soporta 2×200k + working set).

**Riesgos**: el margen OOM por GPU EMPEORA (todos los pesos + TODO el
drafter en una tarjeta — el OOM del 27-oct fue precisamente el drafter sin
sitio; a TP1 la GPU1 tendría aún menos margen que los 391 MiB del test
262k).

**Veredicto A**: **NO recomendado para rendimiento** (por request la mitad;
agregado igual). Solo tiene sentido si algún día se quiere **aislamiento
multi-tenant real** (2 modelos distintos, 2 contextos independientes, 2
fallos aislados — p. ej. un Qwen3.8 en GPU0 y un segundo modelo en GPU1), y
asumiendo margen OOM más estrecho.

---

## B) Qwen3.8-Flash-Next (Maximilian228/NVFP4-GPTQ-Strata)

**Hallazgo bloqueante**: la ficha HF lo dice explícitamente — *"Only for
Strata NVFP4. The files are Strata's runtime formats... **They do not load
in transformers, vLLM or llama.cpp**"*. Es un **runtime de Windows**
(strata.exe) diseñado para CPU+GPU híbrido (1 tarjeta RTX 20-50 + 64-128 GB
RAM, expertos en host). **No es servible por vLLM — descartado tal cual.**

**El modelo base (Qwen/Qwen3.8-Flash-Next) sí es otra historia** — es un MoE
híbrido de 125B (51B densos): 63,3 GiB de expertos NVFP4 + dense 5,6 GiB +
PLE fp8 47,7 GiB (host) + MTP 0,77 GiB. En V100s upstream lo sirve así:
- PR #1242: **Flash-Next IQ3_S GGUF, TP4 en 4×V100-16GB** (64 GB totales,
  expert cache DMA-staged), MTP4, 2-8 concurrentes, ventana 1M.
- #1277: DeepSeek-V4.1-Flash 4×V100 512K packed KV.

**En nuestro rig (2×32GB = 64 GB totales, TP2)**:
- NVFP4 (63,3 GiB de expertos): 31,7 GiB por GPU → **no cabe** (32 GB
  totales por GPU menos pesos dense/KV/activaciones). Necesitaría TP4.
- IQ3_S GGUF (~48-52 GiB): 24-26 GiB por GPU en TP2 → **al límite**, con
  calidad 87-90% vs Q8_0 (caída real) — solo evaluable con el stack de la
  campaña.
- **Bloqueo duro en cualquier caso**: el soporte Flash-Next (QSA, PLE,
  prefill staging, expert cache) vive en upstream main (ola 22-oct, #1242),
  NO en nuestra base → exige la campaña upstream → **bloqueada por #1269**
  (el kernel roto que ya reportamos con repro).

**Veredicto B**: el repo Strata concreto es incompatible con vLLM; el
Flash-Next real es un modelo de **clase TP4/4-GPU en V100** (o Strata en
Windows con 64-128 GB RAM), y su ruta hacia nosotros pasa por: 1) fix
upstream #1269 (nuestro reporte ya publicado acelera esto), 2) campaña
upstream, 3) elegir cuant GGUF que quepa en TP2 — con calidad inferior a
nuestro NVFP4 actual. **No recomendado a corto plazo**; nuestro Qwen3.8-27B
denso NVFP4 sigue siendo el modelo óptimo para 2×V100.

---

## Resumen ejecutivo

| Opción | Vel. por request | Agregado | Contexto | Riesgo | Veredicto |
|---|---|---|---|---|---|
| TP2 actual (fork/main rescatado) | **175 t/s** | 109 t/s | 200k | bajo (parches validados) | **ÓPTIMO — en prod** |
| 2× TP1 | ~85-90 t/s | ~109 t/s (igual) | 200k/instancia | OOM margen peor | solo multi-tenant aislado |
| Flash-Next Strata (repo indicado) | — | — | — | **incompatible con vLLM** | descartado |
| Flash-Next GGUF IQ3_S TP2 | ? | ? | ? | exige campaña upstream + calidad −10% | bloqueado por #1269 |

**Recomendación**: mantener la prod actual (que hoy está a paridad de
baseline con todos los fixes), seguir el comentario upstream #1269, y
reintentar la campaña upstream cuando lo arreglen — ahí es donde Flash-Next
y la línea MTP concurrente se abren de verdad, previsiblemente con una
segunda máquina (o 2 GPUs más) para el TP4 que Flash-Next pide en V100.
