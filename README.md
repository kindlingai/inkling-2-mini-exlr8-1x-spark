# Inkling-2-Mini EXLR8 + Melded DFlash2 on 1× DGX Spark

Single-box (`TP=1`) launch recipe for serving **[kindlingai/inkling-2-mini-exlr8-k2-k3-k4](https://huggingface.co/kindlingai/inkling-2-mini-exlr8-k2-k3-k4)** on one **NVIDIA DGX Spark (GB10)** with **[kindlingai/gr8](https://github.com/kindlingai/gr8)** (`PR #615`, commit `6b874803`+).

One single pack includes:
- **Bundled 279 MB melded DFlash2 drafter** (`dflash2-melded.safetensors`, vs 945 MB BF16) — automatically discovered by `gr8` with zero extra flags (`--mtp 7`, **8/8 exact greedy parity** vs `--mtp 0`, `1.41×–5.18×` decode speedup).
- **Load-time width selection (`--variant`)** across `k4`, `bit325`, `bit250`, and E2E logit-KL distilled `k2` (`-11.65%` KL vs stock `k2` with zero extra bytes).
- **Blackwell MXFP8 prefill & fast attention** (`3,457–3,484 tok/s` 8K cold prefill; `0.000000` decode KL tax with `GR8_INK2_MX8_DENSE=2`, `GR8_INK2_MX8_LM_HEAD=0`).

---

## Quick Start (Docker)

1. Download the single unified model pack:

```bash
hf download kindlingai/inkling-2-mini-exlr8-k2-k3-k4 \
  --local-dir ./models/inkling-2-mini-exlr8-k2-k3-k4
```

2. Configure and launch (default `VARIANT=k4`, OpenAI API on port `8100`):

```bash
cp .env.example .env
docker compose up -d
docker compose logs -f
```

3. Query the OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a thread-safe Python LRU cache with TTL."}],
    "max_tokens": 512,
    "temperature": 0.0
  }' | jq -r '.choices[0].message.content'
```

## Bare-Metal Build & Launch

```bash
git clone https://github.com/kindlingai/gr8.git
cd gr8
GR8_WIDTHS=1 MODEL=ink2 scripts/build.sh out/bin
# Run with any preset: VARIANT=k4 | bit325 | bit250 | k2
GR8_ROOT=$PWD G6M4=$PWD/out/bin/g6m4_ink2 \
  MODEL_DIR=/path/to/inkling-2-mini-exlr8-k2-k3-k4 \
  VARIANT=k4 bash /path/to/serve.sh
```

## Choosing a Preset (`VARIANT`)

| Preset | Device Image | 40×2K FP64 KL (vs NVFP4 `0.01441`) | 1,362-Task Eval Pooled | 8K Cold Prefill | Best For |
| --- | ---: | ---: | ---: | ---: | --- |
| **`k4`** *(default)* | `18.72 GB` | **`0.010460` (`-27.4%`)** | **`93.54%`** *(ties W4A4, `p=0.6655`)* | `3,457 tok/s` | Maximum quality & smaller than Official NVFP4 (`19.53 GB`) |
| **`bit325`** | `16.43 GB` | `0.015019` (`-2.45%` vs stock) | `92.80%` *(91.5% HE+, 97.8% MATH)* | **`3,484 tok/s`** | Balanced daily driver; 8/8 exact parity |
| **`bit250`** | `14.13 GB` | `0.030018` (`-3.81%` vs stock) | — | `3,480 tok/s` | High-speed coding/math (`111` Code / `133` Math tok/s) |
| **`k2`** *(speed sweep)* | `12.60 GB` | `0.052425` (**`-11.65%`** E2E KL) | — | `3,475 tok/s` | Max throughput (`101` Code, `168` Math, `230` C=6, `480` 6× Math) |

For the `k2` speed-sweep profile, set in `.env`:

```bash
VARIANT=k2
GR8_INK2_MX8_DENSE=1
GR8_INK2_MX8_LM_HEAD=1
GR8_S2_ADAPT=1
GR8_INK2_S2_COST=14.0,2.5,2.0,0.4,0.0,0.0
```

For `128K` single-stream context, set `CAP=139264` in `.env` (`3,472 tok/s` at 8K, `3,150 tok/s` at 32K, `2,732 tok/s` at 64K, `2,162 tok/s` at 128K on `bit325`).

---

## Verified Benchmarks (1× NVIDIA DGX Spark)

### 1. Matched 40×2K (`81,920`-Token) FP64 KL vs BF16

| Model / Variant | Device Image | Mean KL | Median KL | Top-1 | PPL (`ref=5.3840`) | ΔPPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`exlr8` Ship `k4`** | 18.72 GB | **0.010460** | **0.001632** | **96.91%** | **5.3787** | **-0.0053** |
| Official NVFP4+MXFP8 (W4A4) | 19.53 GB | 0.014407 | 0.002640 | 96.27% | 5.3919 | +0.0079 |
| **`exlr8` Ship `bit325`** | 16.43 GB | 0.015019 | 0.003027 | 96.00% | 5.3933 | +0.0093 |
| **`exlr8` Ship `bit250`** | 14.13 GB | 0.030018 | 0.007401 | 93.93% | 5.4365 | +0.0525 |
| **`exlr8` Ship `k2`** (E2E KL) | 12.60 GB | **0.052425** | 0.015119 | **91.70%** | 5.5250 | +0.1410 |

### 2. Controlled Real-World Task Evaluation (`1,362` Frozen Tasks × 5 Contenders = `6,810` Completions)

Execution- and SymPy `1.14.0`-verified at `temp=0.0`, `seed=42`, `max_tokens=6144`, `reasoning_effort="low"`, `0` errors, `0` prompt-token mismatches:

| Contender | Weight Size | HumanEval+ (164) | GPQA-Diamond (198) | MATH-500 (500) | GSM8K-500 (500) | Pooled (1,362) | Non-Trunc Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unsloth `UD-Q8_K_XL` (W4A8) | 22.87 GB | 93.29% (153) | 80.30% (159) | 99.00% (495) | 96.00% (480) | 94.49% (1287) | 95.05% |
| Unsloth `UD-Q8_K_XL` (W4A4) | 22.87 GB | 91.46% (150) | 78.79% (156) | 98.80% (494) | 96.20% (481) | 94.05% (1281) | 95.10% |
| `gr8` Official Base (W4A4) | 19.53 GB | 92.07% (151) | 76.26% (151) | 98.00% (490) | 97.20% (486) | 93.83% (1278) | 95.80% |
| **`gr8` `exlr8 k4` (W4A16)** | **16.20 GB** | 91.46% (150) | 75.25% (149) | 98.00% (490) | 97.00% (485) | **93.54%** (1274) | **95.79%** |
| **`gr8` `exlr8 bit325` (W3.25A16)** | **13.91 GB** | 91.46% (150) | 72.22% (143) | 97.80% (489) | 96.40% (482) | **92.80%** (1264) | **94.89%** |

- `exlr8 k4` vs Official Base W4A4 (same `gr8` engine): statistical tie (`-0.29 pp`, exact McNemar `p = 0.6655`; exact tie on MATH-500 `98.00%`).
- `exlr8 k4` vs Unsloth `UD-Q8_K_XL` W4A4: not statistically significant (`-0.51 pp`, `p = 0.3916`), while `6.67 GB` smaller and `2–3×` faster on `gr8` + DFlash2.
- `exlr8 bit325`: holds parity on HumanEval+ (`91.46%`) and MATH-500 (`97.80%`, Level 4–5 `97.33%`), with a documented `-4.04` to `-6.57 pp` drop on GPQA-Diamond (`72.22%`, `p = 0.0241` vs Unsloth W4A4).

### 3. 1× DGX Spark Throughput (Zero-Flag Melded DFlash2 `K=7`)

| Metric | Ship `k4` | Ship `bit325` | Ship `bit250` | Ship `k2` Sweep |
| --- | ---: | ---: | ---: | ---: |
| Code / Math / JSON / HTML (tok/s) | 64.6 / 107.3 / 90.9 / 77.4 | 90.6 / 105.2 / 91.2 / 85.3 | 111.3 / 133.4 / 108.3 / 100.0 | **100.5 / 167.5 / 148.9 / 144.1** |
| Prose / Narrative / Summary / Counting | 48.6 / 42.6 / 55.2 / 162.8 | 49.5 / 46.6 / 61.8 / 172.4 | 55.3 / 46.1 / 66.0 / 186.3 | **80.8 / 74.1 / 97.4 / 228.3** |
| C=6 Mixed Agg Decode (tok/s) | 139.1 | 160.0 | 167.6 | **230.0** (`479.5` 6× Math) |
| Thinking-ON Prose `c=1/2/4/6` (`t=0.6`) | — | 48.5 / 70.2 / 100.3 / 122.6 | — | **66.9 / 98.0 / 150.5 / 166.3** |
| 8K Cold Prefill (tok/s) | 3,457 | **3,484** | 3,480 | 3,475 |
| 3-User 1K Avg TTFT | 0.935s | 0.935s | 0.942s | **0.827s** |

---

## Attribution & License

- Recipe scripts & configuration: Apache-2.0
- Base model: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0)
- Melded DFlash2 drafter source: [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (`4f9088ea66c21f2b99ed657d1941c306b03e84d2`, Apache-2.0)
- Inference engine: [kindlingai/gr8](https://github.com/kindlingai/gr8)
