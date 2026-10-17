# Local Model Bench

A simple bash script to benchmark LLMs running locally on a GPU via [Ollama](https://ollama.com). 
It sweeps a model across a range of context window sizes, measuring generation speed, 
time-to-first-token, and GPU memory usage — and stops automatically if the model 
starts spilling over into system RAM.

> ⚠️ Don't blame me if this script eats your GPU... you've been warned.

Refer to our [blog post comparing and tuning AI models with this script](https://bitsnbytes.dev/posts/2026-10/comparison-and-tuning-local-ai.html)
for a writeup on how to interpret the results.

## NVidia Support

NVidia cards are **NOT** supported at the moment.
If you have an nvidia card and would like to help, feel free to submit a PR
with the fixed `nvidia-smi` commands herein.

## What it does

For each model, [`bench.sh`](bench.sh) does the following:

1. **Checks dependencies** — requires `jq`, `curl`, `ollama`, and either `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD). It also refuses to run if another Ollama model is already loaded.
2. **Collects GPU info** — reads the card name and total VRAM from `nvidia-smi` or `rocm-smi`.
3. **Records model info** — runs `ollama show <model>` and captures the model's maximum context length.
4. **Sweeps context sizes** — generates a list of context sizes (up to 32 steps) based on the GPU's VRAM:
   - ≤ 8 GB VRAM → steps of 1024
   - ≤ 16 GB → steps of 4096
   - ≤ 24 GB → steps of 8192
   - > 24 GB → steps of 16384
   - The sweep stops early if a context size exceeds the model's max context.
5. **Runs the benchmark** — for each context size, it sends the prompt *"Write a 'hello world' application in Python"* to Ollama's `/api/generate` endpoint (non-streaming) with `num_ctx` set to the current context size, then extracts:
   - `tokens_per_second` (eval tokens / eval duration)
   - `time_to_first_token_ms` (prompt eval duration)
   - token count and total eval duration
6. **Checks memory** — records `ollama ps` output and the GPU memory used by the `llama-server` process. If the model starts using CPU/GPU (i.e., offloading to system RAM), it halts the run for that model to avoid thrashing and diminishing results.
7. **Saves results** — appends a row to `data/benchmark-data.csv` and writes full details to `data/benchmark-<model>.log`.

## Usage

```bash
# Benchmark a single model
./bench.sh qwen3.5:9b

# Benchmark all models listed in ollama
./bench.sh all

# No argument: lists available models
./bench.sh
```

## Output

### `data/benchmark-data.csv`

One row per context-size run, suitable for graphing:

```csv
Card,Date Time,Model,Context,Max Context,Memory (MB),Max Mem (MB),Tokens,Duration (ms),Tokens/sec,Offload
Radeon RX 7900 XTX,20261016-1139,qwen3.5:9b,512,262144,6437,24560,358,3988.91,89.75,
Radeon RX 7900 XTX,20261016-1139,qwen3.5:9b,1024,262144,6437,24560,274,3048.91,89.87,
Radeon RX 7900 XTX,20261016-1139,qwen3.5:9b,1536,262144,6437,24560,201,2241.06,89.69,
```

| Column | Description |
|---|---|
| `Card` | GPU name (e.g. `Radeon RX 7900 XTX`) |
| `Date Time` | Run timestamp (`YYYYMMDD-HHMM`) |
| `Model` | Ollama model name |
| `Context` | Context window size used for this run |
| `Max Context` | Model's maximum context length |
| `Memory (MB)` | GPU memory used by the model process |
| `Max Mem (MB)` | Total GPU VRAM |
| `Tokens` | Number of tokens generated |
| `Duration (ms)` | Total generation time in milliseconds |
| `Tokens/sec` | Generation speed |
| `Offload` | `x` if the model started using system RAM (run halted) |

### `data/benchmark-<model>.log`

A human-readable log containing, for each context size:

- Model information (`ollama show` output)
- The full JSON response from Ollama (including the generated text, thinking, and timing metrics)
- `ollama ps` process info
- GPU memory usage

Example excerpt from `data/benchmark-qwen3.5-9b.log`:

```
Starting benchmark for qwen3.5:9b at 20261016-1139...
Model information:
  Model
    architecture        qwen35
    parameters          9.0B
    context length      262144
    quantization        Q4_K_M

===================================================================
Running model at 512 context length...
===================================================================
{
  "model": "qwen3.5:9b",
  "response": "# Hello World in Python\n\n## Basic Version\n\n```python\nprint(\"Hello, World!\")\n```...",
  "eval_count": 358,
  "eval_duration": 3988913000,
  "tokens_per_second": 89.75,
  "time_to_first_token_ms": 73.93
}

-------------------------------------------------------------------
Process information:
NAME          ID              SIZE      PROCESSOR    CONTEXT    RUNNER      UNTIL
qwen3.5:9b    56671c2ab938    5.5 GB    100% GPU     2048       llamacpp    4 minutes from now

-------------------------------------------------------------------
GPU memory usage:
Memory used: 6.3GiB
```

## Requirements

- Linux with an ~~NVIDIA~~ or AMD GPU
- [Ollama](https://ollama.com) with at least one model pulled
- `jq`, `curl`
- `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)
