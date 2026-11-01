# AI inference infrastructure: cheatsheet

Terms from most basic to most advanced. Each has a short definition, an
everyday analogy, and, where possible, where you've already seen it in
this project (🔬).

**Levels**
1. [How a model works](#level-1-how-a-model-works)
2. [Generating text](#level-2-generating-text)
3. [Measuring performance](#level-3-measuring-performance)
4. [Hardware basics](#level-4-hardware-basics)
5. [Inside a serving engine](#level-5-inside-a-serving-engine)
6. [Serving APIs and streaming](#level-6-serving-apis-and-streaming)
7. [Routing and gateways](#level-7-routing-and-gateways)
8. [Making models smaller and faster](#level-8-making-models-smaller-and-faster)
9. [Operating in production](#level-9-operating-in-production)
10. [Scaling beyond one GPU](#level-10-scaling-beyond-one-gpu)
11. [Advanced serving techniques](#level-11-advanced-serving-techniques)

---

## Level 1: How a model works

### Model
A file of numbers plus the code that uses them to turn input text into a prediction of the next piece of text.
> **Analogy:** a very well-read autocomplete. It has read a library and guesses what comes next.

### Inference
Using a trained model to produce outputs. (Training is building the model; inference is using it.)
> **Analogy:** training = a chef learning recipes for years; inference = cooking a dish when an order comes in.
> AI *infra* for inference is about running the kitchen: many orders, fast, cheaply.

### LLM (large language model)
A model trained on huge amounts of text to predict the next token. GPT, Llama, Qwen, Claude.

### Token
The unit a model reads and writes: a word, part of a word, or punctuation. ~1 token ≈ ¾ of an English word.
> **Analogy:** Lego bricks. Text is built from a fixed set of bricks; "unbelievable" might be "un" + "believ" + "able".
> 🔬 Your 1,000-token synthetic prompt was ~5,100 characters.

### Tokenizer
Converts text ↔ token IDs (integers). Each model family has its own, so the same text gives different token counts on different models.
> 🔬 That's why the client reads token counts from the server's `usage`, not by counting characters.

### Vocabulary
The full set of tokens a tokenizer knows. Typically 32k–150k entries.

### Parameters / weights
The learned numbers inside the model. "3B" = 3 billion parameters. More parameters ≈ more knowledge, but bigger and slower.
> **Analogy:** the strength of connections in a brain; learned during training, fixed during inference.
> 🔬 Llama-3.2-3B at 4-bit = 1.7 GB file; Qwen2.5-0.5B = 276 MB.

### Bias
A learned number added after a multiplication, letting a neuron shift its output. Counted among the parameters.

### Neural network / layer
A stack of mathematical steps (layers). Each layer transforms the data a bit and passes it on.
> **Analogy:** an assembly line; each station refines the product.

### Embedding
Turning a token ID into a list of numbers (a vector) that captures its meaning. Similar words get similar vectors.
> **Analogy:** GPS coordinates for words: "king" and "queen" are close, "banana" is far away.

### Hidden size
The length of each token's vector inside the model (e.g. 896 for Qwen 0.5B, 3,072 for Llama 3B).

### Transformer
The architecture behind almost all LLMs: a stack of identical layers, each with an *attention* step and a *feed-forward* step.

### Attention
The step where each token looks at all earlier tokens and decides which matter for predicting the next one.
> **Analogy:** reading "The trophy didn't fit in the suitcase because **it** was too big". To understand "it", you look back and focus on "trophy".

### Q, K, V (query, key, value)
Three vectors computed for each token inside attention:
- **Query:** "what am I looking for?"
- **Key:** "what do I contain?" (used to match queries)
- **Value:** "what information do I hand over if picked?"
> **Analogy:** a library. Your question is the *query*, book titles are *keys*, book contents are *values*.

### Attention head
Attention is split into several parallel "heads", each learning to focus on different patterns (grammar, names, numbers…).
> **Analogy:** several readers each highlighting different things in the same text.

### Feed-forward network (FFN / MLP)
The second part of each transformer layer: processes each token on its own. Holds most of the parameters.

### Logits
The model's raw score for every token in the vocabulary as the possible next token.
> **Analogy:** a scoreboard with 150,000 candidates; the highest score is the model's top guess.

### Softmax / probabilities
Turns logits into probabilities that add up to 1.

### Context window / context length
The maximum number of tokens (prompt + output) the model can handle at once.
> 🔬 You ran prompts up to 16k tokens; Qwen2.5 supports 32k, Llama 3.2 128k.

### Prompt / completion
The prompt is what you send; the completion is what the model generates.

### System prompt
Instructions placed before the user's message ("You are a helpful assistant").
> 🔬 Qwen's template adds "You are Qwen, created by Alibaba Cloud…": that's the ~25 tokens always cached.

### Chat template
The model-specific format that wraps a list of messages (system/user/assistant) into one prompt string with special tokens.

### Special tokens
Tokens with control meaning, not text: start of message, end of message, end of sequence.

### Instruct / chat model vs base model
A base model just continues text; an instruct model was further trained to follow instructions and chat.

---

## Level 2: Generating text

### Autoregressive generation
Generating one token at a time, each based on all previous tokens, then feeding it back in.
> **Analogy:** writing a sentence word by word, re-reading what you've written before choosing the next word.

### Forward pass
One run of the input through all the model's layers to produce logits.

### Sampling
Choosing the next token from the probabilities.

### Greedy decoding
Always pick the highest-probability token. Deterministic.
> 🔬 Your experiments used temperature 0 (greedy).

### Temperature
Controls randomness. 0 = always the top choice; higher = more adventurous choices.
> **Analogy:** 0 = always ordering your usual dish; 1.5 = picking at random from the menu.

### Top-k / top-p (nucleus) sampling
Only sample from the k most likely tokens (top-k), or from the smallest set whose probabilities add to p (top-p). Cuts off silly choices.

### Seed
A number that fixes the random choices so results can be reproduced.

### EOS (end of sequence)
The special token meaning "I'm done". Generation stops when the model emits it.
> 🔬 Counted in `usage` but not streamed: 3 tokens, 2 chunks.

### max_tokens
The maximum number of tokens to generate. A ceiling, not a target.

### finish_reason
Why generation stopped: `stop` (EOS or stop word) or `length` (hit max_tokens).
> 🔬 `--fixed-length` banned EOS so every request ended with `length`.

### Logit bias
Adding to or subtracting from specific tokens' scores before sampling, to encourage or ban them.
> 🔬 That's how `--fixed-length` works: EOS gets −100.

### Prefill
Processing the whole prompt in one parallel pass to build the KV cache and produce the first token.
> **Analogy:** reading the whole question before you start answering.
> 🔬 Prompt length drives TTFT: 129 ms → 29 s for 100 → 16k tokens on Llama 3B.

### Decode
Generating output tokens one at a time after prefill.
> **Analogy:** writing the answer word by word.
> 🔬 ~10 ms per token on Llama 3B, ~100 tokens/s.

### KV cache
Stored keys and values of all previous tokens, so they don't have to be recomputed for every new token.
> **Analogy:** notes in the margin. Instead of re-reading the whole book for each new sentence, you check your notes.
> 🔬 Size per token = 2 × layers × KV heads × head dim × bytes: 12 KB (Qwen 0.5B), 112 KB (Llama 3B).

### Prompt cache / prefix cache (concept)
Keeping KV caches from earlier requests so a new request starting with the same text skips that part of prefill.
> **Analogy:** a teacher who already read the first 10 pages of an essay doesn't re-read them when the student hands in a revised version.
> 🔬 28–110× faster TTFT on a repeated prompt.

---

## Level 3: Measuring performance

### Latency
Time taken for something to happen. For LLMs there are several kinds (below).

### TTFT (time to first token)
Time from sending the request to receiving the first token. Mostly prefill time (+ queueing).
> **Analogy:** how long until the waiter says the first word of the reply.

### TPOT (time per output token)
Average time between output tokens after the first. Measures decode speed.

### ITL (inter-token latency)
The individual gaps between consecutive tokens. TPOT is their average; ITL shows the spikes.
> **Analogy:** TPOT = average typing speed; ITL = the pause between each keystroke.

### E2E latency (end-to-end)
Total time from request to last token. ≈ TTFT + (output tokens − 1) × TPOT.

### Throughput
Work done per second: output tokens/s, or requests/s, across all users.
> **Analogy:** latency = how fast one customer is served; throughput = customers served per hour.

### Latency vs throughput trade-off
Serving more requests together raises throughput but usually makes each one a bit slower.

### Percentiles: p50, p90, p95, p99
p95 = 95% of requests were faster than this. p50 = median. Tail percentiles (p99) show the worst user experiences.
> **Analogy:** a pizza place that delivers in 20 min on average but takes 90 min for 1 in 100 orders has a bad p99.

### Mean vs median
The mean is pulled by outliers; the median isn't. Use medians and percentiles.

### Coefficient of variation (CV)
Standard deviation ÷ mean: how noisy a measurement is.
> 🔬 Llama decode CV 1.6% (steady); Qwen 0.5B 14% (noisy).

### Warm-up
Initial requests excluded from measurement because the first runs are slower (loading, compiling, caches empty).

### SLO (service level objective)
A target, e.g. "TTFT p95 < 500 ms and TPOT p95 < 50 ms".

### SLA (service level agreement)
An SLO written into a contract, with penalties if missed.

### Goodput
Throughput counting only requests that met the SLO.
> **Analogy:** a factory making 100 parts/hour where 30 fail inspection has goodput 70.

### Closed loop vs open loop
- **Closed loop:** N users, each waits for a reply before sending the next request. Load backs off when the server slows down.
- **Open loop:** requests arrive on a schedule regardless of how the server is doing. Like real traffic.

### Coordinated omission
A measurement bug in closed-loop tests: when the server stalls, the client stops sending, so the stall is under-counted.
> **Analogy:** measuring checkout wait times only for customers who didn't give up and leave.

### Poisson arrivals
Random arrival times with a fixed average rate. A standard model of real users.

### Saturation
The load beyond which throughput stops growing and latency shoots up.
> **Analogy:** a motorway at rush hour: adding cars no longer adds throughput, only queues.

### Concurrency
How many requests are in progress at the same moment.
> 🔬 All your Milestone 2 tests used concurrency 1.

---

## Level 4: Hardware basics

### CPU vs GPU
CPU: few powerful cores, good at sequential logic. GPU: thousands of simple cores, good at doing the same math on lots of data at once.
> **Analogy:** CPU = a few professors; GPU = a stadium of students each doing simple sums in parallel.

### Accelerator
Any specialised AI chip: NVIDIA GPU, AMD GPU, Google TPU, AWS Inferentia, Apple GPU.

### VRAM / HBM (high bandwidth memory)
The GPU's own fast memory. Holds weights + KV cache. Its size limits model size and concurrency.
> **Analogy:** the size of the kitchen counter: everything you work on has to fit on it.

### Unified memory (Apple silicon)
CPU and GPU share one pool of RAM. No separate VRAM.
> 🔬 Your Mac's 48 GB is shared; "GPU memory" shows up as `IOAccelerator (graphics)` in `footprint`.

### Memory bandwidth
How fast data can be read from memory (GB/s). Sets decode speed, because every token reads all weights.
> **Analogy:** the width of the pipe between the warehouse and the factory.
> 🔬 M4 Pro ≈ 273 GB/s → Llama 3B floor ≈ 6.2 ms/token; you got 9.9 ms (~63%).

### FLOPS
Floating-point operations per second: raw compute speed. Sets prefill speed.

### Compute-bound vs memory-bound
- **Compute-bound:** waiting on arithmetic (prefill: many tokens at once share each weight read).
- **Memory-bound:** waiting on data from memory (decode: one token per full weight read).
> **Analogy:** compute-bound = chefs are the bottleneck; memory-bound = chefs are idle waiting for ingredients.

### Arithmetic intensity
Math operations per byte read. High = compute-bound; low = memory-bound. Batching raises it during decode.

### MFU / MBU (model FLOPS / bandwidth utilisation)
The share of the hardware's peak compute (MFU) or bandwidth (MBU) you actually use.

### CUDA
NVIDIA's platform for programming GPUs. Most AI software depends on it.

### Metal / MLX
Apple's GPU API (Metal) and Apple's ML framework that uses it (MLX).

### Kernel
A function that runs on the GPU (e.g. a matrix multiply or attention). Speed often comes from better kernels.

### Kernel launch overhead
Fixed cost each time the CPU starts a GPU kernel. Matters when the work per kernel is tiny.
> 🔬 Why Qwen 0.5B ran at 3.4 ms/token instead of its ~1 ms bandwidth floor: overhead-bound, CPU at 100%.

### Precision / data types: FP32, FP16, BF16, FP8, INT8, INT4
Bits per number. FP16/BF16 are standard for inference; FP8/INT8/INT4 are smaller and faster at some quality risk.
> **Analogy:** measuring with a ruler marked in millimetres (FP32) vs centimetres (INT4): less precise but much lighter to carry.

### Model memory rule of thumb
Weights ≈ parameters × bytes per parameter. 8B model × 2 bytes (FP16) = 16 GB. At 4-bit ≈ 4–5 GB.

---

## Level 5: Inside a serving engine

### Inference server / serving engine
Software that loads a model and serves requests over HTTP: vLLM, SGLang, TensorRT-LLM, TGI, llama.cpp, mlx-lm.

### vLLM
The most widely used open-source LLM serving engine. Known for PagedAttention and continuous batching.

### Batch / batching
Processing several requests together in one forward pass. Reading the weights once serves many requests → much higher throughput.
> **Analogy:** a bus instead of individual taxis: one trip, many passengers.

### Static batching
A fixed batch runs until every request in it finishes. Short requests wait for the longest one.
> **Analogy:** a bus that waits at the last stop until the last passenger gets off before picking anyone new up.

### Continuous batching (in-flight / iteration-level batching)
After every token step, finished requests leave and new ones join.
> **Analogy:** a bus where people hop on and off at every stop.

### Scheduler
The part of the engine that decides, each step, which requests run, which wait, and which get paused.
> **Analogy:** a restaurant host deciding who gets a table.

### Waiting / running queues
Requests waiting to start vs being processed. Waiting count = a key load signal.

### Padding / attention mask
Filling shorter sequences to the same length in a batch and telling attention to ignore the filler.

### Memory fragmentation
Wasted memory from reserving big contiguous chunks that aren't fully used.
> **Analogy:** a car park where each car reserves a bus-sized space just in case.

### PagedAttention / paged KV cache
Storing the KV cache in small fixed-size blocks anywhere in memory, tracked by a table. Almost no waste, so many more requests fit.
> **Analogy:** an OS's virtual memory pages; or a library storing a book's chapters on any free shelf, with an index card saying where each chapter is.

### Block / block size
One fixed-size chunk of KV cache (e.g. 16 tokens' worth).

### Block table
Per-request list mapping its token positions to physical blocks.

### Block allocator
Hands out and reclaims free blocks.

### Preemption
Pausing a running request and freeing its KV memory when space runs out. It's resumed later (recomputed or swapped back).
> **Analogy:** an airline bumping a passenger to the next flight.

### Swapping vs recomputation
Two ways to resume a preempted request: copy its KV to CPU memory and back (swap), or recompute it from the tokens (recompute).

### KV cache utilisation
Percentage of KV memory in use. Near 100% → preemptions and queueing.

### Prefix caching (automatic, block-level)
Engine hashes KV blocks by their content; a new request reuses matching blocks.
> 🔬 On a hit, `cached_tokens` = prompt − 1: the last token is always recomputed to get the first output token.

### Reference counting
Counting how many requests use a shared block, so it's only freed when nobody needs it.

### Eviction / LRU
Removing old cache entries when memory is needed. LRU = least recently used goes first.
> 🔬 Probably why E4's cache-miss runs were slow after the 16k-token prompts.

### Chunked prefill
Splitting a long prompt's prefill into chunks and mixing them with other requests' decode steps, so one long prompt doesn't freeze everyone.
> **Analogy:** a cashier serving quick customers between scanning a huge trolley.
> 🔬 mlx-lm's `--prefill-step-size 2048`.

### Head-of-line blocking
One slow item at the front of a queue delays everything behind it.

### max-num-seqs / max-model-len / gpu-memory-utilization
Key vLLM knobs: max concurrent sequences, max context length, and share of GPU memory vLLM may use.

### CUDA graphs
Recording a sequence of GPU kernels once and replaying it, cutting launch overhead. Big win for small batches.

---

## Level 6: Serving APIs and streaming

### OpenAI-compatible API
The de facto standard HTTP API (`/v1/chat/completions`, `/v1/models`) that most servers implement.

### Streaming
Sending tokens as they're generated instead of waiting for the full reply.

### SSE (server-sent events)
The streaming format: `data: {...}` lines over one HTTP response, ending with `data: [DONE]`.
> 🔬 Milestone 1.

### Chunk
One SSE event, usually carrying one token's text in `choices[0].delta.content`.

### Keepalive
Comment lines (`: keepalive`) sent to stop connections timing out while nothing is generated yet.
> 🔬 Must be skipped or TTFT is wrong.

### usage
The server's token counts: `prompt_tokens`, `completion_tokens`, `cached_tokens`. Requires `stream_options.include_usage` when streaming.

### Request timeout / cancellation
Limits on waiting, and stopping generation if the client disconnects (otherwise the GPU keeps working for nobody).

---

## Level 7: Routing and gateways

### Replica
One running copy of the model server. Multiple replicas serve more users.

### Load balancer
Spreads requests across replicas.

### Reverse proxy
A server that receives client requests and forwards them to backends (Envoy, NGINX).

### Upstream / backend / endpoint
The servers a proxy forwards to.

### Health check
Periodically checking that a backend is alive; remove it from rotation if not.

### Round-robin
Send requests to replicas in turn: 1, 2, 3, 1, 2, 3…
> Bad for LLMs: a 10-token request and a 10,000-token request count the same.

### Least connections / least load
Send to the replica with the fewest active requests (or shortest queue, or most free KV cache).
> **Analogy:** choosing the supermarket checkout with the shortest queue.

### Session affinity / sticky sessions
Keep a user's requests on the same replica.

### Consistent hashing
Mapping keys (e.g. session IDs) to replicas so that adding or removing a replica moves only a few keys.
> **Analogy:** seats around a round table. A new guest sits between two others and takes only some of one neighbour's dishes.

### Prefix-aware / KV-cache-aware routing
Send a request to the replica most likely to hold its prompt prefix in KV cache.
> **Analogy:** sending a returning patient to the doctor who already has their file open.
> 🔬 The payoff you measured: 28–110× faster TTFT on a hit.

### API gateway / AI gateway
A front door for model APIs: auth, rate limits, routing between models/providers, usage tracking. Envoy AI Gateway, LiteLLM, Kong AI.

### Envoy
A high-performance proxy; the basis of Istio and Envoy AI Gateway.

### Listener / cluster / filter (Envoy)
Listener = where Envoy accepts traffic; cluster = a group of upstream endpoints; filter = a processing step on a request.

### ext_proc (external processing)
An Envoy filter that sends request/response data to your own service, which can modify it or choose the backend.

### Gateway API
Kubernetes' standard API for configuring traffic routing (successor to Ingress).

### Gateway API Inference Extension
Adds LLM-aware routing to Gateway API.
- **InferencePool:** a group of model-server pods.
- **Endpoint picker (EPP):** the component that picks the best pod using load and cache info.

### Token-based rate limiting
Limiting users by tokens consumed, not request count, since requests vary hugely in cost.

### Retries / fallback
Retry on another replica if one fails (only before streaming starts). Fall back to another model if the first is unavailable.

### Model routing
Picking which *model* serves a request (cheap model for easy prompts, big model for hard ones).

---

## Level 8: Making models smaller and faster

### Quantisation
Storing weights (and sometimes activations or KV cache) in fewer bits. Smaller and faster, sometimes slightly less accurate.
> **Analogy:** compressing a photo to JPEG: much smaller, usually you can't tell.
> 🔬 Both your models were 4-bit.

### Weight-only vs weight-and-activation quantisation
Only shrink the stored weights (helps memory-bound decode), or also the math (helps compute-bound prefill too).

### GPTQ / AWQ
Popular methods for 4-bit weight quantisation that keep quality high.

### FP8
8-bit floating point; supported natively on newer NVIDIA GPUs (H100, L40S). A common production choice.

### KV cache quantisation
Storing the KV cache in 8 or 4 bits so more requests and longer contexts fit.

### Distillation
Training a small model to imitate a big one.

### Pruning / sparsity
Removing unimportant weights.

### FlashAttention
An attention kernel that avoids writing the large attention matrix to memory. Much faster, especially for long contexts.

### GQA / MQA (grouped-query / multi-query attention)
Several query heads share one set of K/V heads, shrinking the KV cache.
> 🔬 Llama 3B: 24 query heads but 8 KV heads; Qwen 0.5B: 14 query heads but 2 KV heads.

### Accuracy evaluation
Checking that optimisations didn't hurt quality, e.g. with lm-evaluation-harness benchmarks (MMLU, GSM8K).

---

## Level 9: Operating in production

### Observability
Being able to tell what the system is doing from the outside: metrics, logs, traces.

### Metrics / Prometheus / Grafana
Numeric time series (metrics), the system that collects them (Prometheus), and dashboards (Grafana).

### Counter / gauge / histogram
Metric types: counter only goes up (total requests); gauge goes up and down (queue length); histogram records distributions (latency buckets).

### DCGM / DCGM exporter
NVIDIA's GPU monitoring: utilisation, memory, power, temperature, errors.

### GPU utilisation (and why it misleads)
"100% utilised" only means a kernel was running, not that the GPU was working efficiently. Look at MFU/MBU too.

### Tracing (OpenTelemetry)
Following one request through all components with timings for each hop.

### Kubernetes
Runs containers across a cluster: scheduling, restarts, scaling, networking.

### GPU Operator / device plugin
NVIDIA's components that let Kubernetes see and schedule GPUs (`nvidia.com/gpu: 1`).

### Node selector / taints and tolerations
Ways to make GPU pods land on GPU nodes and keep other pods off them.

### MIG (multi-instance GPU) / time-slicing
Splitting one GPU into isolated slices (MIG) or sharing it by taking turns (time-slicing).

### Autoscaling: HPA / KEDA
Adding/removing replicas based on load. For LLMs, scale on queue depth or KV usage, not CPU.

### Cold start
Time for a new replica to be useful: pull image (GBs) → load weights (GBs) → warm up. Can be minutes.
> **Analogy:** opening a new checkout lane takes 10 minutes; by then the rush may be over.

### Capacity planning
Working out how many GPUs you need for expected traffic at your SLO.

### Cost per million tokens
GPU cost per hour ÷ tokens per hour × 1,000,000, measured *at your SLO*, the standard way to compare deployments.

### Spot / preemptible instances
Cheaper cloud GPUs that can be taken away at short notice.

---

## Level 10: Scaling beyond one GPU

### Why multiple GPUs
A model too large for one GPU's memory, or more throughput than one GPU can give.

### Data parallelism (DP)
Full copies of the model on different GPUs, each serving different requests. (= replicas)
> **Analogy:** opening more identical restaurants.

### Tensor parallelism (TP)
Each layer's matrices split across GPUs; they compute together and exchange results every layer. Needs very fast links.
> **Analogy:** four chefs each chopping a quarter of every ingredient for the same dish, passing bowls constantly.

### Pipeline parallelism (PP)
Different layers on different GPUs; data flows through them in sequence.
> **Analogy:** an assembly line: GPU 1 does layers 1–20, GPU 2 does 21–40.

### Expert parallelism (EP)
For MoE models: different experts on different GPUs.

### MoE (mixture of experts)
Model with many "expert" sub-networks; a router inside the model picks a few experts per token. Huge total parameters, but only a fraction used per token.
> **Analogy:** a hospital with many specialists; each patient sees only the two they need.

### Collective communication: all-reduce, all-gather
Operations where GPUs combine or share partial results. TP does these every layer.

### NCCL
NVIDIA's library for fast GPU-to-GPU communication.

### NVLink / NVSwitch
Very fast direct links between GPUs in one server (much faster than PCIe).

### InfiniBand / RoCE / RDMA
Fast networking between servers; lets one machine read another's memory directly.

### Interconnect bottleneck
When GPUs spend time waiting on each other's data rather than computing.

---

## Level 11: Advanced serving techniques

### Speculative decoding
A small fast "draft" model guesses several tokens; the big model checks them all in one pass and keeps the correct prefix.
> **Analogy:** a junior writes a draft paragraph; the senior approves most of it in one read instead of writing word by word.

### Acceptance rate
The share of draft tokens accepted. High rate → big speed-up; low rate → wasted work.

### EAGLE / Medusa / n-gram speculation
Variants of speculative decoding that use extra prediction heads or text lookup instead of a separate draft model.

### Prefill/decode (P/D) disaggregation
Running prefill and decode on different servers (or GPU pools), then transferring the KV cache. Each phase gets hardware tuned to it, and long prefills stop disturbing decodes.
> **Analogy:** a restaurant where one kitchen preps and another cooks to order.

### KV cache transfer / KV connectors (NIXL, LMCache)
Moving KV caches between GPUs or servers, or offloading them to CPU memory or disk.

### KV cache offloading / tiered caching
Keeping cold KV blocks in CPU RAM or SSD instead of throwing them away.

### KV cache events
Engines announcing which prefixes they hold, so routers can make exact cache-aware decisions (used by llm-d).

### llm-d
Kubernetes-native distributed inference built on vLLM: cache-aware scheduling, P/D disaggregation.

### RadixAttention (SGLang)
Storing cached prefixes in a tree (radix tree) for efficient sharing across many requests.

### Structured output / constrained decoding
Forcing output to match a format (valid JSON, a schema) by masking invalid tokens at each step.

### LoRA / multi-LoRA serving
LoRA = a small add-on fine-tune for a base model. Multi-LoRA = one base model serving many adapters, swapped per request.
> **Analogy:** one actor, many costumes.

### Priority scheduling / fairness
Giving some users or request types precedence; stopping one heavy user from starving others.

### Long-context techniques
Approaches for very long prompts: sliding window attention, ring attention, context parallelism.

---

## One-page summary

```
Text → tokenizer → tokens → [PREFILL: whole prompt at once, compute-bound, builds KV cache]
     → first token (TTFT) → [DECODE: one token at a time, memory-bandwidth-bound, reads weights + KV cache]
     → tokens streamed over SSE (ITL / TPOT) → EOS or max_tokens → done (E2E)

Engine:  continuous batching + scheduler + paged KV cache + prefix cache   (more users per GPU)
Router:  pick the replica with the right cache and spare capacity          (less recomputation)
Scale:   replicas (DP) → split the model (TP/PP/EP) → split the phases (P/D)
Measure: TTFT, TPOT, ITL, p95/p99, goodput at an SLO, cost per million tokens
```
