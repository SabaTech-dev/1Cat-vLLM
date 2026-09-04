"""Sonda de determinismo de prefill (patron DocAI, 2026-08-28).

Mismo prompt x N veces, temperature=0, max_tokens=1, top_logprobs=20,
una request por vez; compara bit a bit la lista top-20 (token, logprob).
Si no es bit-identica entre corridas, el prefill del stack es no
determinista (bug de kernel), aunque la suite de calidad parezca
estable. Ejecutarla SIEMPRE antes de aceptar una receta de serving
nueva (backend, kv dtype, envs, graphs).

Uso: python f2_determinism_probe.py <url_base> <modelo_served> [n=10]
Requiere el server ya levantado (vllm serve).
"""

import json
import sys

import requests


def main() -> None:
    base = sys.argv[1]
    model = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    prompt = (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "Explain the process of photosynthesis in plants and how light "
        "energy is converted into chemical energy stored as glucose."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": 20,
    }
    runs = []
    for _ in range(n):
        r = requests.post(f"{base}/v1/completions", json=payload, timeout=120)
        r.raise_for_status()
        lp = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
        runs.append(json.dumps(lp, sort_keys=True))
    unique = len(set(runs))
    print(f"corridas: {n} | top-20 unicos: {unique}")
    if unique == 1:
        print("DETERMINISTA: prefill bit-identico entre corridas")
    else:
        print(
            "NO DETERMINISTA: el prefill varia entre corridas — bug de "
            "kernel; investigar antes de servir"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
