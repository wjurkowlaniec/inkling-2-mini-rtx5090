# Basalt 0.5.7 experiment on RTX 5090

Measured 2026-11-25 on one RTX 5090, using the same OpenUncensored checkpoint as the SGLang deployment. This is an experimental comparison, not a replacement of the repository's SGLang setup or a comprehensive quality evaluation. Basalt uses the model's DSpark, while SGLang uses Inco AI's DFlash2. Basalt's converter rounds the checkpoint's attention and other dense projections to Q8_0 and uses FP8 KV; SGLang uses BF16 KV. Consequently these are different numerical implementations, and output differences are expected. Native NVFP4/MXFP8 expert codes are preserved by the converter. The Basalt pack contains vision and the built-in draft. Both tests used temperature 0 and two configured slots with Vision enabled.

Basalt binary: official `v0.5.7`, source tag `v0.5.7`, commit `400b411`. The installed SGLang image and DFlash2 revision are specified in the main README. The Basalt pack was converted from the local uncensored safetensors using the Basalt v0.5.7 converter with 4 workers. The ephemeral pack is not included in this repository. Basalt was served with `--parallel 2 --slot-full-context --max-context 131072 --kv fp8 --vision dynamic`; each Basalt slot is configured for 131,072 tokens, whereas SGLang's 139,264 KV tokens are shared across both requests. This difference matters under large simultaneous workloads.

## Reproducing the Basalt setup

This separately converts the chosen checkpoint; it requires extra storage roughly the size of the model and working memory. Use a free GPU and stop the existing server only when interrupting active requests is acceptable. This is the tested command shape, with local paths chosen by the operator:

```bash
git clone --branch v0.5.7 --depth 1 https://github.com/jesdga95/basalt.git
python -m pip install -r basalt/requirements.txt
python basalt/tools/convert/convert.py --checkpoint /path/to/Inkling-2-Mini-Uncensored \
  --out /path/to/inkling-uncensored.basalt --workers 4
# Use the matching v0.5.7 official release binary, with CUDA 13 installed.
/path/to/basalt-v0.5.7-linux-x64/bin/basalt serve --model /path/to/inkling-uncensored.basalt --alias inkling \
  --host 127.0.0.1 --port 6022 --parallel 2 --slot-full-context \
  --vision dynamic --max-context 131072 --kv fp8
```

Our exact test ran the official `basalt-v0.5.7-linux-x64` archive in the SGLang container image for its CUDA/Python environment. Release archive SHA-256 `a390c775c41d54454ca21c1e547855bf6c251732533ca387e8b890e538090525`; converted file SHA-256 `927c5779e5d0181c7468a612225cdc33a77183467a4d8914c5c6aee25a4c35d5`. Reproducing this file hash requires the exact same input checkpoint and converter. The conversion deliberately changes dense projection precision; it is not a bit-identical checkpoint copy.

## Identical chat prompts, natural EOS

`compare_chat.py` sends an identical short English job-queue prompt for one request; for two requests it simultaneously sends that prompt and a Python LRU implementation prompt. Three batches, 1024 maximum output tokens, streaming, reasoning effort none. All requests in the comparison reached their 1024-token cap, so these are not completed-answer or quality measurements. The prompts and synthetic outputs are in the JSON files. Generated text differs between engines and the client sees reasoning as well as answer tokens. The first Basalt request includes a cold startup wait; later requests reuse a short prefix. SGLang also had a warmed server.

| Server | One request: median estimated decode | One request: output / total time | Two requests: combined output / batch wall time |
| --- | ---: | ---: | ---: |
| SGLang + DFlash2 | 285.2 tok/s | 278.1 tok/s | 469.0 tok/s |
| Basalt + DSpark | 387.9 tok/s | 385.6 tok/s | 493.5 tok/s |

Decode estimate is completion tokens divided by first visible chunk to end of stream; it is not kernel-only throughput and includes parser/transport overhead. Aggregate is all output divided by whole-batch wall time including prefill. The single-prompt estimate is about 36% higher for Basalt; the paired aggregate is about 5% higher. These results do not establish an average across tasks or equal quality. See [SGLang one](chat-sglang-c1.json), [SGLang two](chat-sglang-c2.json), [Basalt one](chat-basalt-c1.json), [Basalt two](chat-basalt-c2.json).

Reproduce the chat test against a compatible local endpoint, when idle:

```bash
python benchmarks/basalt/compare_chat.py --url http://127.0.0.1:6022 --concurrency 1 --output chat-c1.json
python benchmarks/basalt/compare_chat.py --url http://127.0.0.1:6022 --concurrency 2 --output chat-c2.json
```

## Context and function checks

- With 130,214 input tokens including an image, Basalt returned the correct synthetic embedded code, red square and blue circle. Total response time was 16.38 s; its server reported prompt reading 13.1 s and 231 output tokens. This is one retrieval test, not a long-context benchmark suite.
- Two independent synthetic 60,038-token inputs submitted together returned their respective keys in 8.64 s. Both completed naturally, with 145 and 136 output tokens. These results cannot be compared by wall time alone to earlier tests that forced 512 tokens.
- A small image identification and a `get_weather` tool call passed. See [long validation](basalt-long-validation.json) and [short smoke](basalt-smoke.json).
- [Context sweep, one request](basalt-context-c1.json) and [two](basalt-context-c2.json): randomized repetitive filler, 1k/16k/30k/60k, two batches per point, up to 512 output tokens. Unlike the original SGLang sweep, EOS is permitted, so completion lengths vary and the tests are not direct wall-time comparisons. A high rate on repetitive text does not imply the same rate on real agent work.

The Basalt engine ran in a temporary container, and SGLang was restored afterwards. These tests do not establish broader coding quality, long-run stability, Hermes interoperability, or lossless equivalence with the original BF16 projections.
