"""Re-evaluacion CUDA Graphs a util 0.85 (GPU dedicada).
Canario: 'La capital de Francia es' (corrupcion conocida a util baja)."""
import sys, json, time
sys.path.insert(0, "/srv/benchmarks/lhu/upstream/1cat-vllm")
from vllm import LLM, SamplingParams

MODE = sys.argv[1]
llm = LLM(model='Qwen/Qwen3-0.6B', dtype='float16',
          attention_backend='TRITON_PAGED', max_model_len=4096,
          gpu_memory_utilization=0.85, enforce_eager=(MODE == 'eager'))

canary = llm.generate(['La capital de Francia es'],
                      SamplingParams(temperature=0, max_tokens=12))[0]
print(f'CANARY[{MODE}]:', canary.outputs[0].text[:60].replace('\n', ' '))

follow = llm.generate(['The capital of France is',
                       'La fotosintesis es el proceso',
                       'Water boils at',
                       'El sistema solar tiene'],
                      SamplingParams(temperature=0, max_tokens=16))
for o in follow:
    print(f'FOLLOW[{MODE}]:', o.outputs[0].text[:50].replace('\n', ' '))

sp = SamplingParams(temperature=0, max_tokens=256, ignore_eos=True)
llm.generate(['Contando: uno, dos,'], sp)
t0 = time.time()
llm.generate(['Contando: uno, dos,'], sp)
t1 = time.time()
tps = 256 / (t1 - t0)
print(f'TPS[{MODE}]: {tps:.1f}')
json.dump({'mode': MODE, 'canary': canary.outputs[0].text,
           'tps_b1': tps}, open(f'/tmp/opencode/greeval-{MODE}.json', 'w'))
print(f'DONE[{MODE}]')
