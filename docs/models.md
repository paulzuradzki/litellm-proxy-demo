# Proxy models → llama.cpp run config

**Recommended models for coding:**

- `local/qwen3.8-27b-5090`: the default. The newer model, and it takes images as well as text. This config measured ~112 tokens per second.
- `local/qwen3.6-moe-5090`: the fastest. It is a MoE model, so each token reads only ~3B of the 35B weights, and it generates at triple-digit tokens per second.

Use the MoE when you want the extra tokens per second; qwen3.8 takes images and runs at the same speed. The rest of the table exists to get more context, more reasoning, or to use the other card.

The three `-engram` entries are the exception: they run on a different llama.cpp fork and the operator starts that backend by hand, so treat them as available on request rather than always up. See [EngramHalo](#engramhalo) below.

<details>
<summary>Primer on serving inference with llama.cpp</summary>

**How the API is served.** [llama.cpp](https://github.com/ggml-org/llama.cpp) runs LLMs on your own hardware. Its `llama-server` binary loads one GGUF model file and answers OpenAI-compatible requests over HTTP: `POST /v1/chat/completions` takes JSON (`model`, `messages`, `max_tokens`) and returns JSON in OpenAI's shape, streaming tokens as they are produced. The LiteLLM proxy in front of this repo forwards each request to the loaded server, so the whole path speaks the OpenAI dialect. [Ollama](https://ollama.com), [vLLM](https://github.com/vllm-project/vllm), and [SGLang](https://github.com/sgl-project/sglang) do the same job; vLLM and SGLang add batching and multi-GPU tensor parallelism for datacenters. This demo is simpler: it serves one request at a time on one GPU and keeps one or a few models loaded at a time.

**Why one model at a time.** Generating a token reads most of the weights from memory, so speed tracks memory bandwidth. A 27-35B model in 4-bit weights takes 16-23 GB before context adds to it.

| Card | Memory | Bandwidth | Build |
|---|---|---|---|
| NVIDIA RTX 5090 | 32 GB VRAM | ~1,790 GB/s | CUDA |
| AMD Strix Halo (Ryzen AI Max) | up to 128 GB system RAM shared with the iGPU | ~256 GB/s | ROCm |

The 5090 reads memory ~7x faster but holds less; Strix Halo holds much more but generates slower. The gap between those two cards drives most of the differences in the table below.

**Flags.** Shared by every chat entry: `-ngl 999 -fa 1 --no-mmap --jinja`.

| Flag | What it does |
|---|---|
| `-m <file>` | which model to load (quant in the filename, e.g. UD-Q4_K_XL) |
| `-c <n>` | context window in tokens; bigger windows cost KV cache memory |
| `--mmproj <file>` | load the vision projector so the entry accepts images |
| `--spec-type draft-mtp` | speculative decoding with the model's built-in MTP head |
| `--spec-draft-n-max <n>` | draft tokens per step; deep values need the `p-min` floor below |
| `--spec-draft-p-min 0.8` | stop drafting when the head's top choice drops under 0.8 confidence |
| `--spec-type draft-dflash` + `--spec-draft-model` | speculative decoding with a separate small drafter model |
| `--cache-type-k/v q8_0` | 8-bit KV cache: half the memory, speed within ~1% |
| `--reasoning on/off` | force the model's `think` block on or off. Left on, short prompts can burn the whole token budget thinking and return nothing |
</details>

A proxy model name is a key in the host's model registry, listed at `<BASE_URL>/v1/models`, in the pi catalog ([../pi/models.json](../pi/models.json)), and in `scripts/demo.py`. Requesting one starts its server on the named GPU: `-5090` (RTX 5090, CUDA) or `-strix` (Strix Halo, ROCm).

| Proxy model | Architecture | Quant | GPU | Key flags |
|---|---|---|---|---|
| `local/qwen3.6-moe-5090` | Qwen 3.6 35B-A3B MoE + MTP | UD-Q4_K_XL | 5090 | `-c 262144`, `--spec-draft-n-max 2`, `--reasoning off` |
| `local/qwen3.6-moe-think-5090` | same MoE, reasoning on | UD-Q4_K_XL | 5090 | `--spec-draft-n-max 2`, `--reasoning on` |
| `local/qwen3.6-dense-5090` | Qwen 3.6 27B dense + MTP | UD-Q4_K_XL | 5090 | `-c 131072` (VRAM ceiling), `--spec-draft-n-max 2`, `--reasoning off` |
| `local/qwen3.8-27b-5090` | Qwen 3.8 27B hybrid attn+SSM + MTP + vision | UD-Q4_K_XL (proj. BF16) | 5090 | `-c 229376`, 8-bit KV cache, `--mmproj`, `--spec-draft-n-max 4`, `--reasoning off` |
| `local/muse-glimmer-5090` | Meta Muse Glimmer 28B dense + vision + DFlash drafter | kquant-dynamic | 5090 | `-c 131072`, DFlash drafter, `--spec-draft-n-max 15`, `--mmproj` |
| `local/nemotron-lightning-5090` | NVIDIA Nemotron 3.5 Lightning 30B-A3B | NVFP4 | 5090 | `-c 524288`, `--spec-draft-n-max 4`, `--reasoning off` (must stay off: short requests return empty) |
| `local/qwen3.6-moe-strix` | Qwen 3.6 35B-A3B MoE + MTP | UD-Q4_K_XL | Strix | `-c 262144`, `--spec-draft-n-max 16 --spec-draft-p-min 0.8`, `--reasoning off` |
| `local/qwen3.6-moe-think-strix` | same MoE, reasoning on | UD-Q4_K_XL | Strix | `--spec-draft-n-max 16`, `--reasoning on` |
| `local/qwen3.6-dense-strix` | Qwen 3.6 27B dense + MTP | UD-Q4_K_XL | Strix | `-c 262144`, `--spec-draft-n-max 16`, `--reasoning off` |
| `local/qwen3.8-27b-strix` | Qwen 3.8 27B hybrid attn+SSM + MTP + vision | UD-Q4_K_XL (proj. BF16) | Strix | `-c 262144`, `--mmproj`, `--spec-draft-n-max 8`, `--reasoning off` |
| `local/muse-glimmer-strix` | Meta Muse Glimmer 28B dense + vision + DFlash drafter | kquant-dynamic | Strix | same GGUFs and flags as the 5090 twin |
| `local/nemotron-lightning-strix` | NVIDIA Nemotron 3.5 Lightning 30B-A3B | Q4_K_M | Strix | `-c 262144`, `--spec-draft-n-max 16`, `--reasoning off` (must stay off: short requests return empty) |
| `local/qwen3.8-flashnext-engram` | Qwen 3.8 Flash-Next + MTP, on the EngramHalo fork | AD-4.27bpw-Q4_K_M | Strix | `-c 131072`, 8-bit KV cache, `--spec-draft-n-max 4 --spec-draft-p-min 0.75`, thinking capped at 8192 tokens |
| `local/qwen3.8-flashnext-engram-notalk` | same model and drafter, no thinking block | AD-4.27bpw-Q4_K_M | Strix | as above, `--reasoning off` |
| `local/qwen3.8-flashnext-engram-deep` | same model, no drafter | AD-4.27bpw-Q4_K_M | Strix | `-c 262144`, no speculative decoding |

**Why the same model runs differently on the two cards.**

- **Context**: the 32 GB card caps the 27B dense model at 131072 tokens and qwen3.8 at 229376 (the 8-bit KV cache is what makes 229376 fit). The 128 GB card runs the full 262144. Nemotron's 512K window only fits on the 5090's NVFP4 file.
- **Speculative draft depth** (`--spec-draft-n-max`): 2 or 4 on the 5090, 16 on Strix. A deep draft needs the `--spec-draft-p-min` confidence floor, and the 5090 keeps the draft shallow because the 32 GB card has less room for the extra head's memory.
- **Quant**: Nemotron runs NVFP4 on the 5090 and Q4_K_M on Strix. NVFP4 is faster on CUDA but decodes slower on the AMD ROCm build, and the 128 GB card has room for the larger Q4_K_M file.
- **Images**: the `qwen3.8-27b-*` and `muse-glimmer-*` entries accept images on both cards; the others are text only.

## EngramHalo

The `-engram` entries do not run on [llama.cpp](https://github.com/ggml-org/llama.cpp) itself. They run on [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp), a fork written for the Strix Halo iGPU, on a newer ROCm than the other `-strix` entries use. It carries three things mainline does not have: a custom top-k kernel, a sparse gather that reads roughly 2.3K selected KV rows past 16K context instead of the whole cache, and a working MTP draft head for this model. On coding prompts the default lane generates 28-30 tokens per second, which matches `local/qwen3.8-27b-strix` from a much larger model.

**Engram table.** Flash-Next keeps a large lookup table of per-layer embeddings, ~26.8 GB in this quant, separate from the weights the model multiplies through. The server memory-maps it and reads rows on demand rather than holding it in RAM (`-lm mmap --tensor-read-lazy on`), so the table stays on SSD at about 1 GB resident. The weights still need ~54.5 GB, which is why these entries only exist on the 128 GB card.

**Two of the three lanes differ only in the thinking block.** `-engram` caps its `think` block at 8192 tokens; `-engram-notalk` turns it off. The cap is there because an uncapped `think` block can spend a short request's whole token budget reasoning and return empty content, which sends coding agents into retry loops.

**`-engram-deep` trades the drafter for context.** MTP is measured working to about 163K tokens and the full 262144 window only without it, so the deep lane drops speculative decoding to reach the wider window. That costs 30-39% of generation speed on coding work, so reach for it only when a prompt does not fit in 131072 tokens.

**Why these lanes are not always up.** This fork and the mainline `-strix` entries share one pool of memory with no coordination between them, and Flash-Next alone claims about 54.5 GB of it. The operator starts the backend when it is wanted rather than leaving it resident. A request to an `-engram` model while the backend is down returns an error rather than starting it.

## Quantization

Weights are stored at fewer bits than they were trained in, so a model that would need 60 GB in 16-bit fits in 20 GB. The trade is a small loss in quality for a big drop in memory, and the drop is what lets these models run on a single card at all.

The quants in use:

| Quant | Family | Bits | Notes |
|---|---|---|---|
| **Q4_K_XL** / **UD-Q4_K_XL** | k-quant | ~4.5 | The workhorse 4-bit here. `UD` means Unsloth's dynamic requant of it: the same base scheme, but the quantizer picks a higher precision per tensor for the layers that matter most, at a slightly larger file. `K` = the k-quant format (mixed 6-bit blocks); `XL` = a quality tier above `M` and `S`. |
| **NVFP4** | NVIDIA 4-bit float | 4 | NVIDIA's 4-bit floating-point quant, fast on CUDA. Nemotron's 5090 file. |
| **Q4_K_M** | k-quant | ~4.5 | The plain 4-bit k-quant (`M` tier). Nemotron's Strix file: NVFP4 decodes slower on ROCm, and Strix has room for this bigger one. |
| **kquant-dynamic** | Meta k-quant | ~4-8 | Meta's own quant of Muse Glimmer: a per-layer mix of k-quant types (the "dynamic" part) rather than one scheme everywhere. |

### Reading a filename: `Qwen3.8-27B-UD-Q4_K_XL.gguf`

| Part | Meaning |
|---|---|
| `Qwen3.8` | the model family and generation |
| `27B` | roughly 27 billion parameters (the total size, before routing) |
| `UD` | Unsloth dynamic requant (see above) |
| `Q4` | quantized to about 4 bits per weight |
| `_K` | the k-quant format (6-bit blocks, better than a flat 4-bit) |
| `_XL` | quality tier: `XL` > `M` > `S`, more headroom for the important tensors |
| `.gguf` | the file format |

A multimodal model ships two files: the main weights (above) and a `mmproj` file. **mmproj** is the vision projector: a small network that turns image pixels into the embedding space the language model reads. It loads with `--mmproj` and is where the image understanding lives. **BF16** is the format that projector file uses (bfloat16, 16-bit floating point): the projector is small, so it is kept at full precision rather than quantized, because degrading it would blur the images. The Qwen 3.8 projector here is `mmproj-BF16.gguf`.

## Glossary

- **Context window** (`-c`): the maximum tokens of prompt plus answer a request may hold.
- **KV cache**: the per-token key and value tensors the attention layers store so they don't recompute earlier tokens. Grows with context and sequence length, so it is the main memory cost after the weights.
- **Quant (quantization)**: storing weights (or the KV cache) at fewer bits, typically 4, to fit a card. See [Quantization](#quantization) for the specific schemes. A 4-bit 30B model weighs ~17-20 GB.
- **Speculative decoding**: the normal way to generate is one token at a time, which is slow because each step reads the whole model. Speculative decoding speeds this up: a cheap "drafter" guesses several upcoming tokens at once, then the full model checks all of them in a single pass. The guesses that pass count as real output, so you get several tokens for the price of one. When the guesses are wrong, the work is wasted, which is why it only pays off on predictable text (refactors, boilerplate) and not on novel writing. `--spec-draft-n-max` sets how many tokens the drafter may guess per step, and `--spec-draft-p-min` the confidence level below which it stops guessing.
- **MTP (multi-token prediction)**: a built-in drafter. Some Qwen and Nemotron checkpoints are trained with an extra small head that already predicts the next token, so they need no separate drafter file; at inference that head does the guessing in speculative decoding. The cost: the head takes VRAM, a deeper draft (higher `n-max`) takes more of it, and the draft is generated in sequence, one token after another, rather than in parallel. On a 32 GB card that is why the draft stays shallow. You trade memory for a large decode-speed gain on a single request.
- **DFlash**: Muse Glimmer's drafter. Unlike MTP, it is a separate small model file (its own GGUF) that guesses tokens for the main model to check, loaded with `--spec-draft-model`. Same idea as speculative decoding, with a standalone file instead of a built-in head.
- **MoE (mixture of experts)**: an architecture that, for each token, activates only a few of many specialist sub-networks instead of all of them. A 35B MoE reads roughly 3B of weights per token, so it generates much faster than a 35B dense model. The whole model still has to sit in memory, so the speed win comes at no extra memory cost.
- **MTP head vs. a separate drafter**: see MTP and DFlash above. The `-engram` lanes load the MTP head from a second file (`-md`) because the published Flash-Next GGUFs strip the head out of the main weights.
- **Hybrid attn+SSM**: Qwen 3.8's design. Most layers keep a fixed-size "running summary" of the conversation (the SSM part) instead of a growing list of every past token, and only a quarter of layers use the usual attention that stores a per-token cache. Long contexts cost much less memory, which is what lets it run a 224K window on 32 GB.
- **Flash attention** (`-fa 1`): a faster way to compute attention that also uses less memory while it runs.
- **mmap / `--no-mmap`**: by default the OS loads the model file from disk bit by bit as it is needed (memory-mapping). `--no-mmap` reads the whole file into RAM up front, so generation is steady and fast at the cost of a bigger one-time memory hit at load.
- **Jinja chat template**: a recipe stored inside the GGUF that turns your messages into the exact prompt format the model was trained on. `--jinja` switches it on. Tool calling and the reasoning toggle both rely on it.
- **Reasoning**: some models write out their intermediate steps in a `think` block before giving the answer. `--reasoning on/off` forces the mode on or off; leaving it on can make a short prompt spend its entire token budget thinking and return nothing.
- **`--mmproj`**: the vision part of a multimodal model (the projector that reads images). It is a separate GGUF you load alongside the main weights; load it and the model accepts images, skip it and the model is text only.
- **GGUF**: the model file format llama.cpp uses. One file holds the weights plus metadata such as the chat template and context length.
- **TTL**: how many idle seconds the host keeps a loaded model before unloading it to free memory (1 hour here).
