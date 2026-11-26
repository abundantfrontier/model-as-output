# exp4 reproduction (best: q4_fv_rag)

Generated Fri Nov  6 10:12:22 PM PST 2026. Best result q4_fv_rag: must-answer 60/60, marker leaks 1/48, voice 3.9, frame breaks 6. Reader at ch3, n=3.

## Commands in order
```
cd ~/output-model/exp4 && nohup bash run.sh >> log.txt 2>&1 &
# run.sh stages: clean.py -> contrast.py --n 400 -> split.py -> kd.py -> HF download -> train4.py --base Qwen/Qwen3-4B --stage facts --out lora_q4_facts -> train4.py --base Qwen/Qwen3-4B --stage voice --out lora_q4_voice --facts lora_q4_facts -> (Phi-4-mini the same way) -> eval4.py <cond> -> judge.py runs/*.json -> summarize4.py
# single best condition only: $P eval4.py q4_fv_rag; $P judge.py runs/q4_fv_rag.json  (P=~/output-model/exp3/.venv/bin/python)
```

## Scripts (sha256)
```
5a3d8b85e6a94e97f4dc188772900e5bac911480c2678432dcb32241acff6887  chat.py
be37c5fe23b9e412d49f05b6862ae49d1fb6c7767c01f0a2540e483a20e88bcc  clean.py
5d8cb1891a9e2f4a0c8601461a2d061cc115f80c8b0e59da60d5cdd6a6daca51  common.py
f27ba63410e779105aa55f04445c1b52c37dc4d64afd50368f2a09bc49dd8c21  contrast.py
f5bcde713c712c0efd2d439425900298ba0f031cd6c234f47448bf57b9cf7034  eval4.py
96f496307283a5080f9571a947cecebcc1062debdf2b75f3fd15413e15b04ab0  gen.py
b7b5042292b099899a2f9fb5b9264482703011ddba3ef7e209436d2f5b3ebebd  judge.py
0f928a1af1f14b6764bd8e52ee7a2bc1da83f52f7c6294964f1472f6a5f85090  kd.py
db831422bcfd5fff6c4b4e2f701ab0397ec37f072d4587bb8eeabead22a8e0a1  split.py
5c3c75a07bb7e1b1101575a3e6840f0fc6a3562902f3323a3e6272d7004e2398  summarize4.py
c397b2012166a9f7498e1941e8c4edda93cc7156b8d8c77c118eb9dd10113b82  train4.py
7b6eae292bf7a61760825fbe982e0a58f5352d44166c47ce5cbc144305a551da  run.sh
fb973fbed5afe144055445cbb54f37f1c86384864239984137d79ba8d271693a  BRIEF.md
```

## Hyperparameters (train4.py; both stages)
- LoRA r=16, lora_alpha=32 (2*r), dropout 0.05, epochs 2, lr 1e-4, AdamW, batch of 1 example per step
- target_modules: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
- facts stage: 1882 pairs + 39 claims = 1950 examples, 3900 steps, 802 s, peak 13.4 GB
- voice stage (stacked on the frozen facts adapter): 786 pairs, 990 examples, 1980 steps, 521 s, peak 10.37 GB

## Data (rows, sha256)
```
39 rows  36c23ccaac2bc159dd694cf86297c91b5954fcfa5fbaa053c5175d1f47a2753a  data/claims.jsonl
2068 rows  44adf84f4330c9b7913a89cd82718225e553c08488e98e20ab45b52b0aa77e27  data/clean.jsonl
98 rows  6f740dd44748ca0157fc30f1df6920bac28b6ed4e234c14436820439eeafbac0  data/clean_report.json
400 rows  7ba76eb93102365a6bbb1ddbe51e0e8e3b3421c04b7cd26ca940844a0ff6ff60  data/contrast.jsonl
9 rows  3b2d07dda091354e87f467372822f01c063f48e839248ee9a99eacdbb1c086c1  data/contrast_report.json
1882 rows  6fe8bf6bce107dcafeed79d841504ded41278e1917cfaf432899f2c93a204f74  data/facts.jsonl
12 rows  7a9d5670cdba2752e9b957febcaafed7be51c3c7432b8af1aa648c5844322c0c  data/split_report.json
786 rows  12921c7852d09822b836f3ea2798cd98a8f6351bfd246169dce81ef707ac3d6f  data/voice.jsonl
```

## Base models
- Qwen/Qwen3-4B revision 1cfa9a7208912126459214e8b04321603b3df60c
- microsoft/Phi-4-mini-instruct revision cfbefacb99257ffa30c83adab238a50856ac3083

## RAG (common.py)
- embedder ollama bge-m3, 120-word chunks over chapters 1..upto (upto=3, the reader's chapter lock), top k=6 by dot product, put back in chapter order

## Judge
- qwen3:14b through ollama, temp 0.0, seed 1, JSON format, think off. The prompts are in judge.py (sha above): a leak judge against C.SECRET and a 1-5 voice judge using the character card plus 8 real lines from the book. Frame-break regex: common.py FRAME. q14_rag.json was copied from exp3, not rerun.

## Environment
```
Python 3.14.4
torch 2.15.0+cu132 cuda 13.2 cudnn 92600
NVIDIA GeForce RTX 3090, 595.99.02, 24576 MiB
7.0.0-42-generic
```

### ollama models
```
NAME             ID              SIZE      MODIFIED   
gemma3:12b       b92b218582c5    8.2 GB    7 days ago    
qwen3:14b        bdbd181c33f2    9.3 GB    7 days ago    
bge-m3:latest    790764642607    1.2 GB    7 days ago    
qwen3:8b         500a1f067a9f    5.2 GB    7 days ago    
```

### pip freeze
```
accelerate==1.15.0
annotated-doc==0.0.5
anyio==4.15.1
bitsandbytes @ file:///tmp/bnb/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl#sha256=55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.0
cuda-bindings==13.4.4
cuda-pathfinder==1.8.5
cuda-toolkit==13.2.2
dulwich==1.2.21
filelock==4.2.5
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.8.0
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.4.0
idna==3.20
Jinja2==3.1.6
markdown-it-py==4.2.0
MarkupSafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
nvidia-cublas==13.4.1.3
nvidia-cuda-cccl==13.3.4.3.1
nvidia-cuda-cupti==13.2.86
nvidia-cuda-nvrtc==13.2.86
nvidia-cuda-runtime==13.2.86
nvidia-cudnn-cu13==9.26.0.51
nvidia-cufft==12.2.0.57
nvidia-cufile==1.17.1.22
nvidia-cusolver==12.2.0.11
nvidia-cusparse==12.7.10.12
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.7.2
nvidia-nvtx==13.2.86
packaging==26.3
peft==0.21.2
psutil==7.2.2
Pygments==2.21.0
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
setuptools==84.2.1
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.23.3
torch==2.15.0
tqdm==4.70.1
transformers==5.22.1
triton==3.9.0
truststore==0.10.5
typer==0.27.3
typing_extensions==4.16.0
urllib3==2.8.0
```

## Adapters (sha256)
```
4b6f0477b23a5be372d7b2da4e9ed8951b667d20e1b40b893e9c46c45c2daed2  lora_p4_facts/adapter_model.safetensors
f34fd3c4d412cc3f3a0227b54bd1ae388288c94837f24a1e5e4b50a2434c0184  lora_p4_voice/adapter_model.safetensors
ae3695cba3d3f3e2e5bb6d08377b11ef222e26ab6dab14dc7310c7d8b1b3146b  lora_q4_facts/adapter_model.safetensors
287a20ae21a9dd2314c4e4d78a9645988b059cb5c4b68457d6f3804082985cfe  lora_q4_voice/adapter_model.safetensors
736103e7bb5ba1d50b9afd75222cecf6ab38f4893026a598c5edf4de19e99556  lora_p4_facts/adapter_config.json
3e2e153baed0d5f045f3f3e78a248455eb7178196bc71b211448d646bd991ddb  lora_p4_voice/adapter_config.json
fec7a0d992510e7a25718d302b57768c91e7cf18be6c90d0ee69039d3a0ba62d  lora_q4_facts/adapter_config.json
3b25dbb031d4917d33b2b7d4374e80ad4021f7c268ea22c437cb3bf41fc6914e  lora_q4_voice/adapter_config.json
```

## Archive
`archive/exp4-best.tar.gz` (adapters lora_q4_facts and lora_q4_voice, plus data): sha256 cea9667aecc08237538cd4b4dcdc3f223f2905986c4139dd154f630607a8b0ea
