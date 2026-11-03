# LM_Dec_Base_LLaMA2

LLaMA2 - Base family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (2 seed-repeat runs, see results)**
- Failure class: none

## Provenance

- Source repository: LM_2023_llama2-pytorch
- Upstream / license: rasbt/LLMs-from-scratch Llama 2 conversion code (Apache-2.0); license file missing in repo - add

## Differences from the source ([Unified] protocol)

- hidden_dim 1408 (about 8/3 x dim); dtype float32 weights, bf16 autocast

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.54 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.733 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.460 / test 1.444 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.360 / test 1.359 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 1803 MB, inference 441 MB
- Peak GPU memory at length 8192 (batch 1): train None, inference None
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.25 (keys removed 0.23); 4 pairs @ 2048: 0.21 (keys removed 0.28); 16 pairs @ 512: 0.06 (keys removed 0.07); 16 pairs @ 2048: 0.08 (keys removed 0.07)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.26 (keys removed 0.25); 4 pairs @ 2048: 0.22 (keys removed 0.23); 16 pairs @ 512: 0.99 (keys removed 0.02); 16 pairs @ 2048: 0.02 (keys removed 0.02)
- LM-06 memory update `lm06`: @512: latest 0.51, unchanged 0.02, stale 0.01; @2048: latest 0.26, unchanged 0.05, stale 0.04

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check llama2
python run.py lm02 llama2      # enwik8 is downloaded to data/ on first use
```
