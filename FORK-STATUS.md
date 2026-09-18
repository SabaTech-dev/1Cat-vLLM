# Estado del fork — SabaTech-dev/1Cat-vLLM

> Última actualización: 2026-09-18 (inventario real de trabajo propio)

## NUESTRO TRABAJO EN ESTE FORK (inventario real)

Este fork NO es un espejo pasivo — tiene trabajo propio sustantivo:

- **78 branches de trabajo**: agent/private-v100-dsv4-* (deterministic-fp8-tactics,
  fused-fp16-aux, pp2tp4-followup, prescale-spec-match), agent/v100-audit-* (pr341-glm53,
  pr346-dflash2-quality, pr361-qwen38-82t), codex/v100-dflash2-* (fp8-verify20,
  grouped-q16-guard, labd-adaptive-quality, long-decay-wave, prefill-closure,
  quality-audit), codex/v100-fix-pr344-fp13-default.
- **2 PRs ABIERTOS al upstream** (sin merge, base vieja Aug 31 — requieren rebase a main):
  - #435 build: restore SM70 (Volta) compilation — 3 csrc fixes
  - #431 platform: disable custom all-reduce on SM70 (Volta) — TP2 was hit
    (el MISMO fix que Redhatvale documentó como necesario en PCIe)
- Logs de experimentos en root: SM70_FLASH_V100_QUALITY_EXPERIMENT_LOG_20260615.md,
  SM70_MTP_OUTPUT_QUALITY_AUDIT_20260616.md.

## ACLARACIÓN sobre "1 ahead / 696 behind"

Esa comparación es SOLO de la rama main (snapshot Aug 31 + nuestro FORK-STATUS.md docs).
El trabajo real vive en las 78 branches y los PRs abiertos. main NO refleja el esfuerzo.

## PLAN

1. Rebasar #435 y #431 sobre upstream main (696 commits) y reclamar merge.
2. Sincronizar main de este fork con upstream.
3. Trackear v1.6.0 (contendría #645 Mamba-grid — requisito para híbridos qwen35moe vía vLLM).

## Posición de serving (2026-09-18)

- Prod: v1.5.0 (último release), denso, health estable.
- MoE híbrido vía vLLM: upstream main lo sirve YA (medido: single 91.2/101.0, 6-stream 392.3
  agregados, ctx 32K TP2) — sync de este fork + build desde main el día que se active.
- Benchmarks completos: repo sabatech-sm70-qflash, ROADMAP.md sección SERVING STACK.
