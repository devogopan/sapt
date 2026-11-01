# AI inference infra: learning to-do

Learning by building. About one hour a day. Tick items as you go.

Terms in *italics* are explained in [docs/ai-infra-cheatsheet.md](docs/ai-infra-cheatsheet.md).
Look each one up when you reach it, then write it in your own words in `learning-definitions/`.

| Phase | What | Where | Rough time |
|---|---|---|---|
| 1 | Build a tiny inference engine | Mac | 2–3 weeks |
| 2 | Build a KV-cache-aware router | Mac | 2–3 weeks |
| 3 | Hands-on with vLLM on a real GPU | Rented GPU | 1–2 weeks |
| 4 | Routing on real GPUs + Kubernetes | Rented GPU | 2–3 weeks |
| 5 | Advanced topics (pick what interests you) | Mixed | Open |
| — | Upstream contribution (any time from Phase 2) | — | — |

---

## Phase 1: Tiny inference engine (Mac, 2–3 weeks)

**Goal:** understand what vLLM does inside by building a small version of it.
**Tools:** MLX or PyTorch with MPS. Model: `Qwen/Qwen2.5-0.5B-Instruct` (non-quantised, so the weights are plain).
**Reference when stuck:** nano-vllm (GitHub, ~1,200 lines), Karpathy's "Let's build GPT" video, mlx-lm's `generate.py`.

### 1.1 Load a model and generate one token
Terms: *token, tokenizer, weights, layer, transformer, logits, forward pass*
- [ ] Load the tokenizer; encode and decode a sentence; print the token IDs
- [ ] Load the weights; print layer count, hidden size, heads, KV heads (compare with `config.json`)
- [ ] Run one forward pass on a prompt; take the highest logit; decode it; that's the next token
- [ ] Apply the chat template yourself and look at the system prompt it adds

**Done when:** you can generate one correct next token from your own code.

### 1.2 Generation loop without a KV cache
Terms: *autoregressive, greedy decoding, EOS, context window*
- [ ] Loop: append the new token, run the whole sequence again, repeat until EOS or `max_tokens`
- [ ] Print the time per step. Watch it grow as the sequence grows
- [ ] Compare output with `mlx_lm.generate` at temperature 0. They should match

**Done when:** your loop produces the same text as mlx-lm, slowly.

### 1.3 Add sampling
Terms: *temperature, top-k, top-p, greedy, seed*
- [ ] Implement temperature, top-k, top-p
- [ ] Generate the same prompt at temperature 0, 0.7 and 1.5 and compare

### 1.4 Add a KV cache
Terms: *KV cache, Q/K/V, attention, prefill, decode*
- [ ] Split generation into prefill (whole prompt once) and decode (one token at a time)
- [ ] Store K and V per layer; in decode, feed only the new token
- [ ] Compare speed with and without the cache
- [ ] Work out cache size per token (2 × layers × KV heads × head dim × bytes)

**Done when:** output is identical to 1.2 but much faster, and you can explain why.

### 1.5 Static batching
Terms: *batch, padding, attention mask, throughput vs latency*
- [ ] Run 4 prompts of different lengths together (pad + mask)
- [ ] Try batch 1, 4, 16: total tokens/s rises a lot, per-request speed barely drops (decode is *memory-bound*)
- [ ] Notice the waste: short requests sit idle until the longest one finishes

### 1.6 Continuous batching + scheduler
Terms: *continuous batching, scheduler, waiting/running queues, iteration-level scheduling*
- [ ] Request queue: requests arrive at any time
- [ ] Each step: finished requests leave, waiting requests join (prefill them, then decode with the rest)
- [ ] Wrap it in an OpenAI-compatible HTTP server with SSE streaming
- [ ] Hit it with several `curl` streams at once and watch them interleave

**Done when:** your own server streams correct answers to several clients at the same time.

### 1.7 Paged KV cache
Terms: *PagedAttention, block, block table, block allocator, fragmentation, preemption*
- [ ] Fixed pool of KV blocks (e.g. 16 tokens each); free-list allocator
- [ ] Each request has a block table; attention gathers its K/V from those blocks
- [ ] When the pool runs out: preempt a request (free its blocks, put it back in the queue)
- [ ] See how many more requests fit than with one big contiguous buffer per request

### 1.8 Prefix caching
Terms: *prefix caching, block hashing, reference counting, eviction (LRU)*
- [ ] Hash full blocks by their token content (+ the previous block's hash)
- [ ] New request: reuse matching blocks instead of recomputing them; report `cached_tokens`
- [ ] Ref-count shared blocks; evict least-recently-used free blocks when memory is short
- [ ] Send the same long prompt twice and watch the second one start almost instantly

### 1.9 Optional extensions
- [ ] Chunked prefill: split long prompts so other requests' decode isn't stalled (*chunked prefill*)
- [ ] Speculative decoding with a smaller draft model (*speculative decoding, acceptance rate*)
- [ ] `/metrics` endpoint: running, waiting, KV usage, cache hit rate (useful in Phase 2)

---

## Phase 2: KV-cache-aware router (Mac, 2–3 weeks)

**Goal:** build the layer in front of inference servers that decides which replica serves each request.
**Backends:** 3–4 `mlx_lm.server` instances on ports 8081–8084, or your own engine from Phase 1.
**Known Mac limit:** all replicas share one GPU, so least-load routing can't show much. Cache-hit effects are real.

### 2.1 Plain proxy
Terms: *reverse proxy, upstream, SSE pass-through, health check*
- [ ] OpenAI-compatible proxy (FastAPI or aiohttp) that forwards to one backend and streams SSE through unchanged
- [ ] Health checks: mark a backend down when `/v1/models` fails

### 2.2 Basic strategies
Terms: *replica, round-robin, least-connections, in-flight requests*
- [ ] Round-robin
- [ ] Least in-flight requests (the router counts its own, since mlx-lm has no metrics)
- [ ] Add a response header saying which replica served the request

### 2.3 Traffic generator
Terms: *session, multi-turn conversation, shared prefix, system prompt, open loop*
- [ ] A small script simulating N users, each holding a growing conversation (every turn resends all previous turns)
- [ ] Optional: a long shared system prompt across all users

### 2.4 Affinity and prefix-aware routing
Terms: *session affinity, sticky sessions, consistent hashing, prefix hashing, cache-aware routing*
- [ ] Session affinity: hash session ID → replica (consistent hashing, so adding a replica moves few users)
- [ ] Prefix-aware: hash the start of the prompt → the replica that saw it last
- [ ] Hybrid: prefer the cache-warm replica unless it's overloaded, then fall back to least-load
- [ ] Run the traffic generator against each strategy and compare `cached_tokens` and time to first token

### 2.5 Envoy in front
Terms: *Envoy, listener, cluster, filter, ext_proc, endpoint picker*
- [ ] Run Envoy locally (brew or Docker) with the replicas as a cluster; round-robin through Envoy
- [ ] Turn your routing logic into an external processor (ext_proc) that picks the endpoint
- [ ] Read how the Gateway API Inference Extension's endpoint picker does the same job

### 2.6 Gateway features (optional)
Terms: *token-based rate limiting, retries, fallback, API keys*
- [ ] Per-key token rate limits (count tokens from `usage`)
- [ ] Retry on a dead backend before the first byte; never retry mid-stream
- [ ] Model fallback (big model busy → small model)

---

## Phase 3: Hands-on with vLLM on a real GPU (rented, 1–2 weeks)

**Goal:** see production software on production hardware, and connect it to what you built.
**GPU:** one NVIDIA L4/A10G/L40S/A100 by the hour (RunPod, Lambda, Vast.ai, or a cloud). Shut it down after each session.

### 3.1 Get it running
Terms: *CUDA, driver, HBM, GPU instance*
- [ ] Script: create instance → run vLLM (Docker image) → SSH tunnel to the Mac → tear down
- [ ] Run `nvidia-smi`; find the GPU's memory size and how much vLLM took for weights vs KV cache (in its startup log)

### 3.2 Find your Phase 1 ideas in vLLM
Terms: *continuous batching, PagedAttention, block size, prefix caching, chunked prefill*
- [ ] Read vLLM's startup log: block size, number of KV blocks, max concurrency it computed
- [ ] Find the scheduler, block manager and prefix cache in vLLM's source; compare with your versions
- [ ] Turn prefix caching off and on; send a repeated prompt; see the difference

### 3.3 Observe it
Terms: *Prometheus, Grafana, DCGM, gauge, histogram*
- [ ] Look at vLLM's `/metrics`: running and waiting requests, KV usage, prefix cache hit rate
- [ ] Optional: Prometheus + Grafana + DCGM exporter for a live dashboard

### 3.4 Push it until it breaks
Terms: *saturation, preemption, KV cache utilisation*
- [ ] Use `vllm bench serve` to raise load until requests start waiting
- [ ] Watch KV usage reach 100% and preemptions appear
- [ ] Try `--max-num-seqs`, `--gpu-memory-utilization`, `--max-model-len` and see what changes

### 3.5 Try a quantised model
Terms: *quantisation, FP8, AWQ/GPTQ*
- [ ] Same model in FP16 vs FP8 or INT4: memory used, how many requests fit, speed

---

## Phase 4: Routing on real GPUs + Kubernetes (rented, 2–3 weeks)

### 4.1 Your router on real replicas
- [ ] 2–4 vLLM replicas (one per GPU, or a small model with several per GPU)
- [ ] Point your Phase 2 router at them. Now least-load routing means something too
- [ ] Use vLLM's metrics (queue depth, KV usage) in your router instead of counting yourself

### 4.2 Kubernetes
Terms: *GPU Operator, device plugin, node selector, InferencePool, HPA, KEDA, cold start*
- [ ] Small cluster with a GPU node; NVIDIA GPU Operator
- [ ] vLLM Deployment + Service; your router or Envoy in front
- [ ] Gateway API Inference Extension: InferencePool + endpoint picker; compare with your router
- [ ] Autoscale on queue depth (KEDA); time a cold start (pod start → model loaded → first token)

### 4.3 Try llm-d
Terms: *llm-d, KV-cache events, P/D disaggregation*
- [ ] Deploy llm-d's quickstart; read how its scheduler scores replicas
- [ ] Compare its cache-aware routing with yours

---

## Phase 5: Advanced (pick by interest)

- [ ] **P/D disaggregation:** prefill and decode on separate vLLM instances with KV transfer (*disaggregated serving, KV transfer, NIXL*)
- [ ] **Multi-GPU model:** tensor parallelism on 2 GPUs with a 70B-class model (*TP, PP, NVLink, all-reduce*)
- [ ] **MoE model:** serve a mixture-of-experts model; look at expert parallelism (*MoE, EP*)
- [ ] **Multi-LoRA:** one base model, many adapters; route by adapter (*LoRA, adapter*)
- [ ] **Speculative decoding in vLLM:** draft model / EAGLE; see when it speeds things up
- [ ] **SGLang:** run it next to vLLM; read how RadixAttention differs from your prefix cache (*RadixAttention*)
- [ ] **Failure drills:** kill a replica mid-stream; overfill the KV cache; set too long a context; read the logs and metrics

---

## Running track: upstream contribution

- [ ] Pick a project: vLLM, llm-d, Envoy AI Gateway, Gateway API Inference Extension, SGLang
- [ ] Read CONTRIBUTING.md; build it locally; run its tests
- [ ] Find a "good first issue" or a docs gap you hit yourself
- [ ] Open the PR

---

## Habits

- One small, finished step per session beats a big half-done one
- Before running something, guess what will happen; then check
- Every new term → `learning-definitions/` in your own words
- Keep notes short: a few lines on what surprised you is enough
