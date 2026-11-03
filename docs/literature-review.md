# Literature review: language-aware expert placement for multilingual MoE serving over PCIe

Topic 08, Inference Research Bootcamp. Prepared on 3 November 2026. The target is a 2027 workshop, first choice EuroMLSys 2027 (with EuroSys 2027; six pages; the bootcamp venue list expects the deadline in late February 2027, which still has to be confirmed).

**How the sources were checked.** Every arXiv ID in this review was looked up through the arXiv API on 3 November 2026, and the returned title matched. GitHub repositories were opened through the GitHub API, and every statement about vLLM or SGLang code was read at the tag or commit named in section 6. Model facts come from the config files on Hugging Face. Numbers attributed to a paper are the authors' own, taken from the arXiv abstract unless stated otherwise; we have not reproduced any of them. Venues are given only where the arXiv record or the paper's own comment states them. Anything we could not check is marked "not verified". The same list, with one line per item, is in `references.json` next to this file.

**Corrections to the topic brief.**

- [2] is titled "Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling". Its figure of 0.8% of expert pairs chosen together by 42% of tokens holds for top-2 routing during pretraining on MI300X GPUs, not for inference.
- [5] is titled "Multilingual Routing in Mixture-of-Experts" (ICLR 2026).
- In the arXiv API text of [4], the headline reads "reduce all2all communication data up to 20". The multiplication sign seems to have been lost, so confirm "20x" against the PDF before quoting it.
- The Gnani model on Hugging Face is `gnani/gnani-evon-v3.3-30B-A3B`. Its config declares a Nemotron-H hybrid with 23 Mamba, 23 MoE and 6 attention layers, not a pure transformer.
- The GitHub API reports `deepseek-ai/DeepInfer` as created on 16 October 2026. The brief gives 20 October as the release date.

## 1. Positioning

Expert parallelism pays for every token that leaves its GPU, and on PCIe-only servers that cost is a large share of each MoE step. Two bodies of work bear on how many tokens can stay local. Systems papers place experts that are chosen together on the same GPU (ExFlow, Occult, MoETuner, Sem-MoE, CAP), and the newest ones condition the plan on a coarse workload class: task families in TACG, and domains such as code, math and chat in [4], which reports up to 20x less all-to-all data on multi-node clusters. Interpretability papers show, separately, that multilingual MoEs route tokens by language in their early and late layers and share experts across languages in the middle layers ([5], Chen et al. 2026, RISE). We found no paper that joins the two: none conditions placement on language or script, none uses the layer-wise split to decide which layers to pin and which to replicate, and none measures the effect end to end on PCIe-only GPUs or under code-mixed traffic. The 2026 PCIe papers (ThunderEP, CoMoE) make each transferred byte cheaper but leave placement alone. The claim this paper can make, and that nobody has made, is this: on 4 and 8 PCIe GPUs, a plan that pins the language-specific layers by language and replicates the shared middle layers within a fixed memory budget moves fewer bytes per token and gives lower p99 latency than load-only replication (EPLB) and than a single language-agnostic co-activation plan at the same budget, and it keeps most of that gain when traffic is code-mixed or its language mix shifts. That is one mechanism, three open models and one hardware class, which is the right size for a six-page workshop paper.

## 2. Related work

Each entry gives the exact title, the first author, the year, the venue where one is confirmed, a link, what the work does, and how it relates to our question: **closer** (it attacks nearly the same problem), **complementary** (it solves a neighbouring problem that we can combine with or learn from), or **baseline** (we should run it or re-implement it).

### (a) Expert placement and co-location from activation or affinity patterns

1. **Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference** (ExFlow). Jinghan Yao et al., 2024, arXiv. [arXiv:2401.08383](https://arxiv.org/abs/2401.08383)

   Shows that pre-trained GPT-style MoEs have strong inter-layer expert affinity (the expert a token uses in one layer predicts its expert in the next) and exploits it with an integer program for placement and a context-coherent form of expert parallelism that needs one all-to-all where earlier designs needed two, without fine-tuning. *Relation: closer.* It is the language-agnostic, cross-layer version of our co-selection analysis, and a pooled ExFlow-style plan is the ablation that shows whether conditioning on language adds anything.

2. **Occult: Optimizing Collaborative Communication across Experts for Accelerated Parallel MoE Training and Inference.** Shuqing Luo et al., ICML 2025. [arXiv:2505.13345](https://arxiv.org/abs/2505.13345)

   Calls two experts chosen by the same token "collaborating" and shows that raising the share of collaborations kept on one device speeds up expert parallelism; it has an exact mode and a lossy mode (collaboration pruning) that needs fine-tuning. *Relation: baseline.* Its intra-device collaboration share is a locality metric we should report, and its exact mode is a language-agnostic placement to compare against. We avoid pruning because it changes model outputs.

3. **Advancing MoE Efficiency: A Collaboration-Constrained Routing (C2R) Strategy for Better Expert Parallelism Design.** Mohan Zhang et al., NAACL 2025. [arXiv:2504.01337](https://arxiv.org/abs/2504.01337)

   Analyses routing as collaboration and specialisation between experts, finds that most experts are overly collaborative, which sends tokens to many devices, and proposes a routing strategy that constrains collaboration. *Relation: complementary.* It changes the router and therefore the model, while we keep routers untouched; its collaboration statistics are still a template for our per-language graphs.

4. **MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing.** Seokjin Go and Divya Mahajan, 2025, arXiv. [arXiv:2502.06643](https://arxiv.org/abs/2502.06643)

   Places experts with an integer linear program that weighs token load, communication and computation together, using the observation that tokens routed to an expert in one layer go to a limited set of experts in the next; the target is tail latency caused by load imbalance and communication skew. *Relation: closer.* This is the formulation to extend with per-language traffic weights and a replication budget, and a candidate baseline.

5. **Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling** (Sem-MoE; version 1 was titled "Speculative MoE"). Yan Li et al., ICLR 2026. [arXiv:2503.04398](https://arxiv.org/abs/2503.04398)

   Offline, it clusters experts onto devices by their co-activation for classes of input; online, it moves requests to the device that holds their likely experts (data-parallel attention) or reshuffles tokens (tensor-parallel attention). It is built into SGLang. *Relation: closer.* It is the nearest system design, and language could be one of its input classes, so check in the full paper whether it already uses language. No public code found.

6. **Scaling Multi-Node Mixture-of-Experts Inference Using Expert Activation Patterns** [4]. Abhimanyu Bambhaniya et al., 2026, arXiv. [arXiv:2604.23150](https://arxiv.org/abs/2604.23150)

   Profiles Llama 4 Maverick, DeepSeek-V3-671B and a large Qwen3 MoE on more than 100k expert-activation traces; finds variable load imbalance, expert popularity that shifts by task family (code, math, chat, general) and a strong prefill-decode correlation; then groups micro-batches by workload and places experts for token locality, cutting all-to-all data by up to 20x. *Relation: baseline.* This is the domain-aware method named in the brief. Our version swaps domain for language and multi-node links for PCIe. No public code found, so it has to be re-implemented from the paper.

7. **Beyond Task-Agnostic: Task-Aware Grouping for Communication-Efficient Multi-Task MoE Inference** (TACG). Zhiyao Xu et al., 2026, arXiv. [arXiv:2606.01007](https://arxiv.org/abs/2606.01007)

   Builds co-activation graphs per task family, reweights them so that locality inside a family dominates, gives each expert a primary GPU under exact capacity limits, and replicates a few generic experts that are central for every family; it reports 31.39% lower average communication cost at a Jain fairness index of 0.9975, holding under distribution shift. *Relation: closer.* This is the closest published method. Our paper must differ in more than the conditioning variable: the layer-wise pin-or-replicate rule, code-mixed traffic, and measured latency on PCIe. No public code found.

8. **Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling** [2]. Radha Gulhane, Quentin Anthony and Beren Millidge (Zyphra), 2026, arXiv. [arXiv:2610.09372](https://arxiv.org/abs/2610.09372)

   Finds correlated expert choices early in pretraining, places co-selected experts together, adds a dispatcher that sends each token to each GPU once (up to 58% fewer dispatched rows), and, under sequence parallelism, moves each token to the GPU predicted to hold its next-layer experts (token shuffling); all-to-all time drops 1.16-2.63x in Megatron-LM on MI300X GPUs. *Relation: closer.* It is about training, but the deduplicating dispatcher and the co-selection statistics carry over to inference directly.

The brief's example list also named Lazarus and Prophet. Lazarus exists ([arXiv:2407.04656](https://arxiv.org/abs/2407.04656), now titled "Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models") but is about resilient training. Pro-Prophet ([arXiv:2411.10003](https://arxiv.org/abs/2411.10003)) is a training load balancer. We did not find the original Prophet paper on arXiv and have not verified it. Lina appears in cluster (b2).

### (b1) Expert-parallel serving systems: engines and libraries

1. **DeepSeek-V3 Technical Report**, DeepSeek-AI, 2024, arXiv ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)), with the **EPLB** repository ([github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB), last commit d52c72d on 24 March 2025).

   DeepSeek-V3 is served with redundant copies of heavily loaded experts. EPLB is the open-sourced planner: given estimated per-expert loads, it replicates heavy experts and packs the replicas onto GPUs, with a hierarchical policy that keeps expert groups on one node and a global policy for large EP sizes. *Relation: baseline.* This is the "EPLB-style" load-based replication in the brief. It sees only per-expert load, not which experts are chosen together or by which language.

2. **DeepEP**, DeepSeek-AI, GitHub ([github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP), release v1.2.1 of 16 September 2025, head f6aa38f of 21 October 2026).

   Dispatch and combine kernels for expert parallelism. Its README lists Hopper or newer GPUs, NVLink inside a node and RDMA across nodes as requirements. *Relation: complementary.* It does not apply to the L40S, an Ada-generation GPU without NVLink, which is why the PCIe transports in cluster (c) matter to us.

3. **DeepInfer** [1], DeepSeek-AI, GitHub ([github.com/deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer), created 16 October 2026, head 5537415 of 25 October 2026).

   The engine behind DeepSeek's service, released for DeepSeek-V4.1-Flash on SM100 GPUs and Ascend 950 NPUs, with data-parallel attention and MoE expert parallelism plus EPLB. Its deployment notes give the EPLB settings: for prefill, 4 redundant experts per GPU, a 100-step statistics window and re-assignment every 1,000 steps (slowest-rank to mean MoE load of 1.074 at EP4DP4); for decode, 1 redundant expert per GPU, a 1,000-step window and re-assignment every 10,000 steps (1.090). Weights move over NVLink symmetric memory. *Relation: complementary.* It shows the state of practice, which is periodic, load-only rebalancing on NVLink-class hardware, and its window and interval settings are sensible defaults for our EPLB baseline.

4. **Efficient Memory Management for Large Language Model Serving with PagedAttention** (vLLM). Woosuk Kwon et al., SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180); EP guide at [docs.vllm.ai](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)

   vLLM (v0.32.0, code read at head 190df0b) runs expert parallelism with `--enable-expert-parallel`, rebalances with `--enable-eplb` (window, interval and redundant-expert settings), exposes an `AbstractEplbPolicy` that maps per-layer expert loads to a physical-to-logical expert map, offers `linear` and `round_robin` initial placement, and defaults to the `allgather_reducescatter` all-to-all backend. *Relation: baseline.* It is also one of the two plugin targets.

5. **SGLang: Efficient Execution of Structured Language Model Programs.** Lianmin Zheng et al., 2023, arXiv. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104); EP guide at [docs.sglang.io](https://docs.sglang.io/docs/advanced_features/expert_parallelism)

   SGLang (v0.5.23, code read at head 62f9f3f) records expert distributions (`--expert-distribution-recorder-mode` with `stat`, `stat_approx`, `per_pass` or `per_token`), rebalances with DeepSeek-style algorithms under `--enable-eplb`, and loads a placement file through `--init-expert-location`, either as an explicit `physical_to_logical_map` or as `logical_count` statistics. *Relation: baseline.* It is the shortest path to serving a static language-aware plan.

6. **Tutel: Adaptive Mixture-of-Experts at Scale.** Changho Hwang et al., 2022, arXiv. [arXiv:2206.03382](https://arxiv.org/abs/2206.03382); repository [microsoft/Tutel](https://github.com/microsoft/Tutel), v0.4.1, still updated in October 2026.

   Uses one layout for parameters and data that every parallelism and pipelining choice can share, so the system can switch between them at run time as the MoE workload changes; it reports 4.96x and 5.75x speedups of a single MoE layer on 16 and 2,048 A100 GPUs. *Relation: complementary.* Adaptive execution, but placement by workload class is not its subject.

7. **DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.** Samyam Rajbhandari et al., ICML 2022. [arXiv:2201.05596](https://arxiv.org/abs/2201.05596)

   An end-to-end MoE training and inference system whose inference side reports 7.3x better latency and cost than earlier MoE inference solutions. *Relation: complementary.* It is the reference point for how MoE serving was first organised.

8. **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts.** Trevor Gale et al., 2022, arXiv. [arXiv:2211.15841](https://arxiv.org/abs/2211.15841); repository [databricks/megablocks](https://github.com/databricks/megablocks), v0.10.0, now archived.

   Writes MoE layers as block-sparse matrix products so that no tokens are dropped or padded when routing is imbalanced. *Relation: complementary.* Its kernels absorb imbalance inside one GPU, while placement decides how much imbalance reaches each GPU. The repository is archived, so treat it as background.

### (b2) Expert-parallel serving systems: load balancing and placement papers

1. **Accelerating Distributed MoE Training and Inference with Lina.** Jiamin Li et al., 2022, arXiv. [arXiv:2210.17223](https://arxiv.org/abs/2210.17223)

   For inference, uses the observed pattern of expert selection to schedule resources so that all-to-all transfer sizes and bandwidth stay balanced despite skewed expert popularity; it reports a 1.63x lower 95th-percentile inference time on average on an A100 testbed. *Relation: complementary.* An early tail-latency result for popularity-aware scheduling, with no notion of language.

2. **HarMoEny: Efficient Multi-GPU Inference of MoE Models.** Zachary Doucet et al., 2025, arXiv. [arXiv:2506.12417](https://arxiv.org/abs/2506.12417)

   Moves tokens to under-used GPUs and prefetches experts from host memory; under heavy imbalance it reports 37-70% more throughput and 34-41% lower time to first token than the next-best baseline. *Relation: complementary.* A dynamic balancer that a static language plan would run alongside.

3. **Accelerating MoE Model Inference with Expert Sharding** (MoEShard). Oana Balmau et al., EuroMLSys 2025. [arXiv:2503.08467](https://arxiv.org/abs/2503.08467)

   Splits each expert's matrices by rows and columns across GPUs so that load is balanced exactly without dropping tokens, with up to 6.4x lower time to first token than DeepSpeed on encoder MoEs. *Relation: complementary.* It is also a useful example of the scope of an accepted EuroMLSys MoE paper.

4. **Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens** (METRO). Yanpeng Yu et al., 2025, arXiv. [arXiv:2512.09277](https://arxiv.org/abs/2512.09277)

   Shows that in memory-bound decode, balancing token counts across GPUs raises the number of activated experts per GPU and slows the step, and routes tokens to balance activated experts instead; against EPLB, measured in vLLM on 8 A100 GPUs and in a simulator, it reports 11-22% lower decode latency. *Relation: complementary.* Decode on the L40S is memory-bound too, so our cost model should count activated experts per GPU, not only tokens.

5. **TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across Memory- and Compute-Bound Regimes.** Jie Li et al., 2026, arXiv. [arXiv:2608.13057](https://arxiv.org/abs/2608.13057)

   Models expert time as the larger of a weight-streaming term and a token term and solves dispatch per batch as a makespan problem; integrated into SGLang, it reports 4-6% more throughput and about 15.6% lower p99 latency for Qwen3-235B. *Relation: complementary.* A recent SGLang balancer that reports p99. The code link in its abstract returned 404 on 3 November 2026.

6. **UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing.** Xinming Wei et al., 2026, arXiv. [arXiv:2606.04101](https://arxiv.org/abs/2606.04101)

   Rebalances every micro-batch and layer from the exact post-gating load, using the scale-up links inside rack-scale nodes to move expert state; averaged over training and serving prefill on up to 256 GPUs, it reaches 94.3% of a force-balanced ideal. *Relation: complementary.* It argues that history-based balancers fail under non-stationary load, which is the risk a static language plan carries; its fast re-planning depends on rack-scale links that PCIe servers do not have.

7. **Communication-Aware Placement and Pruning for Efficient Mixture-of-Experts Inference** (CAP). Xiao Shi et al., 2026, arXiv. [arXiv:2607.05116](https://arxiv.org/abs/2607.05116)

   Groups frequently co-activated experts to cut inter-device and inter-node traffic, trades communication against compute load, and can prune routing destinations; it reports 1.23-1.86x higher throughput than DeepSeek EPLB and vLLM's sequential placement. *Relation: baseline.* A language-agnostic co-activation placement that has already been compared with EPLB inside vLLM. No public code found.

8. **Director: Accelerating Distributed MoE Serving via Online Proactive Expert Placement.** Qianli Liu et al., INFOCOM 2026. [arXiv:2607.08782](https://arxiv.org/abs/2607.08782)

   Predicts the expert activation of incoming requests (with a small cascaded predictor or a low-bit model replica), migrates experts during compute-bound phases, and solves placement with a relaxation that has a (1+ε) approximation guarantee; it reports 11-55% lower end-to-end latency than prior placement methods. *Relation: complementary.* Online re-placement driven by prediction; a request's language is a far cheaper predictor than a model replica.

### (c) MoE on PCIe or commodity GPUs, and offloading

1. **Efficient Expert-Parallel Communication on PCIe-Connected Consumer GPUs** (ThunderEP) [3]. Jaehwan Lee et al., 2026, arXiv. [arXiv:2609.40093](https://arxiv.org/abs/2609.40093)

   On consumer GPUs where every inter-GPU transfer passes through CPU memory, replaces NCCL's relayed transfers with DMA-engine copies and cheaper completion polling; integrated into vLLM on RTX 4090 and RTX 5090 systems, dispatch is 2.00x and combine 1.53x faster than NCCL on average, and end-to-end speedup reaches 1.66x. *Relation: complementary.* It makes each byte cheaper and placement removes bytes, so the two should compose, and a reviewer will ask for both. No public code found.

2. **Democratizing MoE inference on commodity GPUs with CoMoE.** Ruwen Fan et al., 2026, arXiv. [arXiv:2610.09424](https://arxiv.org/abs/2610.09424)

   Uses the host as a routing hub on GPUs without peer-to-peer access: a token needed by several GPUs is written to host memory once, and combine is aggregated per token in host buffers; it reports up to 1.46x throughput on RTX 5090 GPUs, approaching NVLink-capable A800 GPUs at 23.4% of the hardware cost. *Relation: complementary.* A transport-level answer to the same PCIe problem; its write-once dispatch is the same deduplication idea as [2]. No public code found.

3. **FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving.** Qingxiu Liu et al., 2026, arXiv. [arXiv:2604.02715](https://arxiv.org/abs/2604.02715)

   Pages expert weights between compressed GPU memory and host memory under a residency budget, on top of vLLM; for Mixtral-8x7B on 2 L40S GPUs, where weight-resident vLLM does not fit, it reports 4.3x the throughput of KTransformers and 29.1% lower average TPOT. *Relation: complementary.* The only MoE serving paper with L40S results that our arXiv search returned, and a reminder that replicas compete with the KV cache for the L40S's 48 GB.

4. **MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs.** Shiyi Cao et al., 2024, arXiv. [arXiv:2411.11217](https://arxiv.org/abs/2411.11217)

   Pipelines CPU compute, GPU compute and weight transfers for batch inference, guided by a hierarchical roofline model; it reports up to 10.3x higher throughput than earlier offloading systems for Mixtral 8x7B on a single 16 GB T4. *Relation: complementary.* The single-GPU alternative to multi-GPU expert parallelism, useful for explaining why we spread experts over 4-8 GPUs instead.

5. **Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference.** Ranggi Hwang et al., 2023, arXiv. [arXiv:2308.12066](https://arxiv.org/abs/2308.12066)

   Introduces a pre-gating function that selects experts ahead of time, so that offloaded expert weights can be moved from CPU to GPU before they are needed. *Relation: complementary.* Evidence that expert choice can be predicted before the router runs, which request-level, language-based steering also relies on.

6. **SiDA-MoE: Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models.** Zhixu Du et al., MLSys 2024. [arXiv:2310.18859](https://arxiv.org/abs/2310.18859)

   Takes a data-aware approach to predicting expert activation so that dormant experts stay in host memory, reporting up to 3.93x throughput and up to 80% GPU memory saving, with a quality drop as low as 1%. *Relation: complementary.* A precedent for using the input data to decide where experts live.

7. **Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models.** Keisuke Kamahori et al., ICLR 2025. [arXiv:2402.07033](https://arxiv.org/abs/2402.07033)

   Chooses between running experts on the CPU and moving their weights to the GPU, depending on which is cheaper for the scenario; code at [efeslab/fiddler](https://github.com/efeslab/fiddler). *Relation: complementary.* A fallback for single-GPU, memory-limited deployments, not for our multi-GPU setting.

8. **MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache.** Leyang Xue et al., 2024, arXiv. [arXiv:2401.14361](https://arxiv.org/abs/2401.14361)

   Traces expert activation per request at batch size one and uses representative traces to drive expert caching and prefetching, reporting 3.1-16.7x lower per-token latency than vLLM, Ollama, DeepSpeed and BrainStorm. *Relation: complementary.* It uses the same unit we do (the request's activation trace), but to manage memory rather than placement across GPUs.

### (d) Routing analysis by language, script and domain

1. **Multilingual Routing in Mixture-of-Experts** [5]. Lucas Bandarkar et al., ICLR 2026. [arXiv:2510.04694](https://arxiv.org/abs/2510.04694)

   With parallel multilingual data, finds language-specific routing in early and late decoder layers and strong cross-lingual routing alignment in middle layers; a language's performance correlates with how closely its middle-layer routing matches English, and steering middle-layer routers toward English-preferred experts gains 1-2% across three models and more than 15 languages. *Relation: closer.* It is the premise of a layer-wise plan. The open question for us is whether the early and late layers are specific enough, and carry enough traffic, to pay for pinning.

2. **Understanding Multilingualism in Mixture-of-Experts LLMs: Routing Mechanism, Expert Specialization, and Layerwise Steering.** Yuxin Chen et al., 2026, arXiv. [arXiv:2601.14050](https://arxiv.org/abs/2601.14050); code at [conctsai/Multilingualism-in-Mixture-of-Experts-LLMs](https://github.com/conctsai/Multilingualism-in-Mixture-of-Experts-LLMs)

   Finds that routing aligns with language families, that high-resource languages lean on shared experts while low-resource languages depend more on language-exclusive experts, and that early and late layers are language-specific while middle layers act as shared capacity. *Relation: closer.* It suggests grouping languages by family (for example Indo-Aryan and Dravidian) when one GPU group per language is too many, and its code is a starting point for the routing atlas.

3. **Unveiling Language Routing Isolation in Multilingual MoE Models for Interpretable Subnetwork Adaptation** (RISE). Kening Zheng et al., Findings of EMNLP 2026. [arXiv:2604.03592](https://arxiv.org/abs/2604.03592)

   Reports that high- and low-resource languages activate largely disjoint expert sets, with a convergence-divergence pattern across depth, and selects language-specific experts in shallow and deep layers with specificity scores and shared experts in middle layers with overlap scores. *Relation: closer.* Its specificity and overlap scores are close to the per-layer statistic we need to decide whether a layer is pinned or replicated.

4. **OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models.** Fuzhao Xue et al., 2024, arXiv. [arXiv:2402.01739](https://arxiv.org/abs/2402.01739)

   Its routing analysis finds that routing decisions depend mostly on token IDs, with little influence from context, and that token-to-expert assignments are fixed early in pretraining. *Relation: complementary.* If routing follows token IDs, script may matter as much as language, so romanised Hinglish and Devanagari Hindi could route differently. The atlas should test this directly.

5. **OLMoE: Open Mixture-of-Experts Language Models.** Niklas Muennighoff et al., 2024, arXiv. [arXiv:2409.02060](https://arxiv.org/abs/2409.02060)

   A fully open MoE (weights, data, code and logs) whose routing analysis reports high expert specialisation, including by domain and by vocabulary. *Relation: complementary.* A ready set of routing measurements to copy for the per-language atlas.

6. **Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model.** Yeskendir Koishekenov, Alexandre Berard and Vassilina Nikoulina, 2022, arXiv. [arXiv:2212.09811](https://arxiv.org/abs/2212.09811)

   Prunes NLLB-200's experts in a language-specific way, removing up to 80% of them without fine-tuning and with negligible quality loss, so that the model runs on one 32 GB GPU; its pruning metrics pick out language-specific experts. *Relation: closer.* Direct evidence that language-specific experts exist and can be exploited at serving time, though in an encoder-decoder translation model.

7. **Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference.** Sneha Kudugunta et al., Findings of EMNLP 2021. [arXiv:2110.03742](https://arxiv.org/abs/2110.03742)

   Compares token-, sentence- and task-level routing in multilingual translation and shows that routing by task lets a small sub-network be extracted per task, with 1.9x higher peak inference throughput than token-level routing. *Relation: complementary.* It builds language-aware serving into training; our models route per token, so we measure what language structure emerges without retraining.

8. **Extracting Small Translation Specialists from LLMs by Aggressively Pruning Experts.** Liu O. Martin et al., EMNLP 2026. [arXiv:2605.28042](https://arxiv.org/abs/2605.28042)

   Prunes experts irrelevant to translation from modern MoE LLMs without training: half of all experts with negligible loss and 70% with minor losses. *Relation: complementary.* Evidence that multilingual capability in current MoE LLMs sits in a separable subset of experts; Lucas Bandarkar, first author of [5], is a co-author.

### (e) Evaluation methodology for MoE serving

1. **Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference.** Haiyang Huang et al., 2023, arXiv. [arXiv:2303.06182](https://arxiv.org/abs/2303.06182)

   Characterises MoE inference for language modelling and machine translation before proposing dynamic gating, expert buffering and expert load balancing. *Relation: complementary.* The characterise-first structure our paper should copy, and one of the few serving studies with a multilingual (translation) workload.

2. **MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems.** Yinsicheng Jiang et al., 2024, arXiv. [arXiv:2412.07067](https://arxiv.org/abs/2412.07067)

   Argues that MoE systems usually trade one of cost, accuracy and performance for the other two, and introduces sparsity-aware utilisation metrics (sparse memory bandwidth utilisation and sparse model FLOPS utilisation). *Relation: complementary.* Use its sparse bandwidth metric to show how close each configuration is to the L40S memory-bandwidth limit.

3. **MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models.** Krishna Teja Chitty-Venkata et al., 2025, arXiv. [arXiv:2508.17467](https://arxiv.org/abs/2508.17467)

   Measures MoE throughput on H100 GPUs across batch size, sequence length, FFN size and expert count, and across optimisations including parallelisation strategies, for Mixtral, DeepSeek, OLMoE and Qwen models. *Relation: complementary.* A template for the sweep axes, and a reference point for how expert parallelism scales on NVLink hardware.

4. **Demystifying the Communication Characteristics for Distributed Transformer Models.** Quentin Anthony et al., 2024, arXiv. [arXiv:2408.10197](https://arxiv.org/abs/2408.10197)

   Logs the communication of GPT-style training under different parallelism schemes and checks the logs against analytical models. *Relation: complementary.* The method we need for all-to-all bytes: count from the router outputs, predict analytically, and check the two against each other.

5. **Joint Effects of GPU Server Topology, Parallelism, and Congestion Control on MoE Inference: A Controlled Simulation Study.** Kaikai Yuan et al., 2026, arXiv. [arXiv:2609.37828](https://arxiv.org/abs/2609.37828)

   Runs 768 deterministic ASTRA-sim and NS-3 simulations over six server topologies, four TP/EP partitions and four network modes; exposed communication is 89.9-95.8% of mean completion time in the feedback-enabled subset, and the best topology depends on how the logical groups are mapped. *Relation: complementary.* A reminder to report how our GPUs sit on PCIe switches and CPU sockets, because the mapping of EP ranks onto that topology can matter as much as the volume.

6. **Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems.** Amey Agrawal et al., 2024, arXiv. [arXiv:2407.07000](https://arxiv.org/abs/2407.07000)

   Shows that TTFT, TBT, TPOT and normalised latency miss parts of user-facing performance and proposes the fluidity-index metric. *Relation: complementary.* Report token-level latency distributions, not only means, alongside p99.

7. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.** Yinmin Zhong et al., OSDI 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

   Optimises serving for goodput, the request rate served within both TTFT and TPOT targets. *Relation: complementary.* We should report goodput at fixed latency targets for each traffic mix, not only peak throughput.

8. **BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems.** Yuxin Wang et al., 2024, arXiv. [arXiv:2401.17644](https://arxiv.org/abs/2401.17644)

   Releases 10.31 million traces from regional Azure OpenAI GPT services over 213 days, covering request burstiness, conversation patterns, response lengths and failures. *Relation: complementary.* A realistic source of arrival times and lengths; it has no language labels, so the language mix has to be layered on top.

## 3. The gap and our claim

- **Language is the conditioning variable nobody has used, and it is known before routing.** TACG conditions placement on task family and [4] on domain; neither uses language, although a request's script and language can be detected from the prompt before the first MoE layer, and [5], Chen et al. and RISE all find strong language structure in multilingual routing. *Claim to defend:* per-language (or per-family) co-selection graphs serve a larger share of token-expert assignments on the token's own GPU than one pooled graph, at the same number of replicas and the same load balance, on all three models.

- **The layer-wise split is a placement rule, not only an observation.** [5] finds language-specific routing in early and late layers only, and RISE and Chen et al. report the same convergence-divergence shape. *Claim to defend:* a plan that pins the language-specific layers and replicates the shared ones beats both whole-model extremes (pin everything, replicate everything) and a pooled co-activation plan on bytes per token and p99 TPOT, and a per-layer specificity score computed from the traces (for example the Jensen-Shannon divergence between languages' expert distributions) chooses the split without hand tuning.

- **The gain holds on PCIe and composes with a better transport.** Placement results so far come from NVLink or multi-node clusters ([4], Sem-MoE, CAP, Zepp) or from training ([2], Cobalt, PlaceMoE), while the PCIe papers change the transport (ThunderEP, CoMoE). *Claim to defend:* on 4 and 8 L40S GPUs, in vLLM or SGLang, the plan lowers bytes per token on PCIe and p99 latency against default EP and EPLB. We report the effect both as the share of assignments served locally and as deduplicated rows dispatched, so the result does not depend on one all-to-all implementation.

- **Code-mixed traffic and a shifting language mix cost little.** We found no study of LLM MoE routing on Hinglish or Tanglish; the code-switching routing work we found is in speech recognition, for example LR-MoE ([arXiv:2307.05956](https://arxiv.org/abs/2307.05956), INTERSPEECH 2023), which routes speech frames to language experts. *Claim to defend:* with code-mixed requests and a drifting mix, the plan keeps most of its gain until a periodic refresh, and the paper states how often the plan must be recomputed (the same knob as EPLB's window and interval).

## 4. Scoop risk (September to early November 2026)

Searches run on 3 November 2026: date-sorted arXiv API queries on expert placement, co-activation, PCIe expert parallelism, all-to-all and multilingual routing; 12 web searches; and GitHub repository searches for code. Nothing we found from this window conditions expert placement on language, and nothing evaluates code-mixed traffic. The items below come closest.

| Date (2026) | Work | What overlaps | How we differ, or what to do |
|---|---|---|---|
| 29 Sep | Cobalt ([arXiv:2609.36959](https://arxiv.org/abs/2609.36959)) | Co-locates frequently co-activated experts on a node and rebalances within the node; 75.74-99.26% less cross-node token traffic | Training on 32 B200 GPUs; no language, no PCIe, no serving latency |
| 30 Sep | ThunderEP [3] ([arXiv:2609.40093](https://arxiv.org/abs/2609.40093)) | Expert parallelism over PCIe, inside vLLM | Transport, not placement. Treat it as a second axis and ask the authors about code |
| 30 Sep | MegaFlux ([arXiv:2610.00671](https://arxiv.org/abs/2610.00671)) | Runtime expert replication under a per-GPU replica budget | Megakernels on B200 GPUs, driven by load skew rather than language |
| 1 Oct | Cross-Lingual Alignment for Decoder-Only Models using MoE Routers ([arXiv:2610.01921](https://arxiv.org/abs/2610.01921)), Bandarkar et al. | Trains routers to align across languages during continual pretraining | A threat to the premise rather than a scoop: models trained this way may route less by language. Measure specificity per model and report gains as a function of it |
| 7 Oct | Expert Coupling [2] ([arXiv:2610.09372](https://arxiv.org/abs/2610.09372)) | Co-selection placement, a deduplicating dispatcher and token shuffling | Pretraining on MI300X GPUs with no language axis. Adopt its dispatcher idea and cite it; an inference follow-up from Zyphra is the most likely way this space closes |
| 7 Oct (v3 19 Oct) | CoMoE ([arXiv:2610.09424](https://arxiv.org/abs/2610.09424)) | MoE inference on PCIe consumer GPUs; writes shared tokens to the host once | Host-centric transport that ignores placement, on GPUs without peer-to-peer access |
| 8 Oct | Zepp ([arXiv:2610.11158](https://arxiv.org/abs/2610.11158)) | Places expert replicas to reduce token communication, with balance as a constraint | Multi-node serving bound by network cards, language-agnostic. A strong baseline if code appears |
| 10 Oct | PlaceMoE, "Accelerating MoE Training through Expert Placement and Replica Coordination" ([arXiv:2610.14798](https://arxiv.org/abs/2610.14798)) | Co-selection-aware expert grouping plus replica allocation by bounded dynamic programming | The formulation is close to ours but for training. Cite it and keep the novelty on language, layers and PCIe serving |
| 13 Oct | FastEEP ([arXiv:2610.18451](https://arxiv.org/abs/2610.18451)) | Elastic resizing of expert-parallel serving groups | Elasticity, not placement |
| 16-25 Oct | DeepInfer [1] | Production EP with load-only EPLB | NVLink and SM100 only; no co-selection or language |

Also seen and judged not to overlap: TopoEP ([arXiv:2609.35481](https://arxiv.org/abs/2609.35481), topology-aware balancing for training), RailWave ([arXiv:2610.03415](https://arxiv.org/abs/2610.03415), multi-rail traffic shaping on top of DeepEP) and the topology simulation study in cluster (e).

The two papers most likely to be extended into our space are older than the window. TACG (May 2026) already does per-class co-activation grouping with shared replication, and Sem-MoE (ICLR 2026) already steers requests to the GPUs that hold their experts. A short follow-up of either that swaps "task" or "input class" for "language" would take much of the novelty. Two things reduce that risk: the per-language, per-layer routing atlas is publishable on its own, and the layer-wise rule plus PCIe measurements are not in either paper. If the workshop's policy allows preprints, post the atlas and the 4-GPU result as soon as they are solid.

To re-check every two weeks, run these queries against `https://export.arxiv.org/api/query?search_query=...&sortBy=submittedDate&sortOrder=descending&max_results=20` and read anything new: `abs:"expert placement" AND abs:inference`, `abs:"co-activation" AND abs:experts AND abs:placement`, `abs:PCIe AND abs:"expert parallelism"`, `abs:multilingual AND abs:"mixture-of-experts" AND abs:routing`, and `abs:"all-to-all" AND abs:"mixture-of-experts"`. ICLR 2027 submissions on OpenReview are also worth a search; we found none on this topic.

## 5. Reading order

**Week 1, in this order**

1. **[5] Multilingual Routing in Mixture-of-Experts (Bandarkar et al., ICLR 2026).** Read it first because the whole layer-wise idea rests on it. Focus on the layer-wise routing analysis and on which layers count as early, middle and late for each model, because our split will be compared with theirs. About 2 hours.
2. **[4] Scaling Multi-Node MoE Inference Using Expert Activation Patterns (Bambhaniya et al., 2026).** This is the systems template: collect traces, find structure, group requests, place experts, count all-to-all data. It is also the baseline we must re-implement, so note exactly how they count all-to-all data and how they group micro-batches. About 2.5-3 hours.
3. **TACG, Beyond Task-Agnostic (Xu et al., 2026).** The closest published method. List each design choice (per-family graphs, reweighting, primary GPU under capacity, shared replication of generic experts) and decide, before writing any code, where our method differs by more than the conditioning variable. About 2 hours.

**Week 2**

4. **[3] ThunderEP (Lee et al., 2026).** What host-staged PCIe traffic costs and how a vLLM integration for PCIe dispatch looks. About 1.5 hours.
5. **Sem-MoE, Semantic Parallelism (Li et al., ICLR 2026).** How request steering under data-parallel attention is done inside SGLang, and whether its input classes already include language. About 2 hours.
6. **[2] Expert Coupling (Gulhane et al., 2026).** Co-selection statistics, the dispatcher that sends each token to each GPU once, and how they count dispatched rows. About 1.5 hours.

Alongside these, spend 1-2 hours on code: SGLang's `python/sglang/srt/eplb/expert_location.py` and `expert_distribution.py`, and vLLM's `vllm/distributed/eplb/policy/abstract.py` and `docs/serving/expert_parallel_deployment.md`.

## 6. Baselines, metrics and reusable code

**Baselines.** Run all of them at the same replication budget, on the same dispatch path, and with the same request steering where steering applies.

- **B0, default expert parallelism.** vLLM with `--enable-expert-parallel` and the `expert_placement_strategy` setting at `linear` (and `round_robin` as a variant); SGLang with `--init-expert-location trivial`.
- **B1, load-based replication (EPLB style).** vLLM `--enable-eplb` with `num_redundant_experts` set to our budget; SGLang `--enable-eplb`; or DeepSeek's `eplb.rebalance_experts` run offline on the same traces. DeepInfer's windows and intervals are sensible defaults.
- **B2, pooled co-activation placement.** One co-selection graph built from all traffic, the same partitioner and the same budget (the ExFlow, Occult and CAP idea). This baseline isolates the value of language.
- **B3, domain-aware placement [4].** Re-implemented from the paper: label requests by domain (code, math, chat, general), group micro-batches by domain, and place for locality.
- **B4, TACG-style class-aware placement without the layer rule.** Language as the class, one policy for every layer. This ablation isolates the layer-wise contribution.
- **Transport axis.** If ThunderEP or CoMoE code appears, run the best of B0-B4 and our plan on top of it as well.

**Metrics**

- All-to-all bytes per token, per layer, split into dispatch and combine. Count both token-expert rows that leave the GPU and deduplicated token-GPU rows, predict them from router outputs and the plan, and check against PCIe transmit and receive counters (for example from `nvidia-smi dmon` or DCGM).
- Locality: the share of token-expert assignments served on the token's own GPU (Occult's intra-device collaboration, and the measure used in [2]).
- Latency: TTFT, TPOT and end-to-end latency at p50 and p99, plus goodput at fixed TTFT and TPOT targets (DistServe), for each traffic mix.
- Throughput: output tokens per second at a fixed latency target.
- Load imbalance: slowest-rank to mean MoE load per layer (DeepInfer's measure; vLLM's `log_balancedness` and SGLang's `expert_balancedness_report_mode` settings log balancedness), and activated experts per GPU (METRO).
- Memory: replica overhead per GPU, using the formula in vLLM's EP guide (MoE layers times bytes per expert times experts plus redundant experts, divided by EP ranks), and the KV-cache capacity it costs.
- Robustness: bytes and p99 over time while the mix shifts, and the time to re-plan and move weights.

**Code, at the versions we checked**

| Component | Version or commit checked | What to reuse |
|---|---|---|
| SGLang | v0.5.23 (2 Nov 2026); code read at 62f9f3f (3 Nov 2026) | `python/sglang/srt/eplb/`: `expert_location.py` (`--init-expert-location` accepts a `.json` or `.pt` file holding `physical_to_logical_map` or `logical_count`), `expert_distribution.py` (recorder with `stat`, `stat_approx`, `per_pass` and `per_token` modes; check whether `per_token` gives the per-token traces we need), `eplb_algorithms/` (`deepseek`, `deepseek_vec`, `elasticity_aware`, each with a hierarchical variant), `lplb_solver.py`, `eplb_simulator/`. `--ep-num-redundant-experts` sets the budget and `--ep-dispatch-algorithm` (`static`, `dynamic`, `fake`, `lp`) chooses among replicas. Model files `sarvam_moe.py`, `nemotron_h.py` and `qwen3_moe.py` contain the expert-location hooks |
| vLLM | v0.32.0 (18 Oct 2026) and v0.33.0rc3 (2 Nov 2026); code read at 190df0b (3 Nov 2026) and the hooks re-checked at v0.32.0 | `vllm/distributed/eplb/` (`eplb_state.py`, `rebalance_execute.py`, `policy/abstract.py`, where `rebalance_experts(weight[layers, experts], ...)` returns a `physical_to_logical_map`). In `vllm/config/parallel.py`, `EPLBPolicyOption` allows only `"default"` and async EPLB rejects other policies, so a new policy needs a small patch to vLLM, not only an external plugin. Model files `sarvam.py`, `nemotron_h.py` and `qwen3_moe.py` implement the EPLB interface |
| DeepSeek EPLB | d52c72d (24 Mar 2025) | `eplb.rebalance_experts` for the offline load-only baseline |
| DeepSeek LPLB | 0490f79 (19 Nov 2025), [github.com/deepseek-ai/LPLB](https://github.com/deepseek-ai/LPLB) | A linear-programming balancer that its authors describe as early research; SGLang's EPLB package also contains an `lplb_solver.py` |
| DeepEP | v1.2.1; head f6aa38f | Not usable on L40S (requires Hopper or newer GPUs, NVLink and RDMA) |
| MoonEP | [github.com/MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP), created 24 Jul 2026 | vLLM lists it as an NVLink backend with dynamic redundant experts; not for L40S |
| Chen et al. routing code | ea1d65b (14 Jan 2026) | Scripts for multilingual routing analysis |
| Tutel, DeepSpeed, MegaBlocks | v0.4.1, v0.19.9 (29 Oct 2026), v0.10.0 (archived) | Background only |

Two practical points follow from reading the code. First, SGLang can serve a static plan with no code changes, by passing our `physical_to_logical_map` (sized for the experts plus the redundant slots) to `--init-expert-location`. That makes it the faster first prototype, with vLLM second. Second, vLLM's default `allgather_reducescatter` backend sends every token to every EP rank, so, as far as we can tell, the bytes it moves do not depend on placement; only the compute balance does. The selective-dispatch backends listed in both engines are documented for NVLink, RDMA or specific GPUs. Measuring the effect of placement on bytes therefore needs a token-level dispatch path that works over PCIe (see question 1 in section 7).

No public code was found, by GitHub repository search on 3 November 2026, for ThunderEP, CoMoE, TACG, Sem-MoE, CAP, Director, Expert Coupling, Cobalt or [4]. TEMPO's linked repository returned 404.

**Models (Hugging Face configs read on 3 November 2026)**

| Model (revision) | Architecture | MoE layers | Routed experts, top-k, shared | Vocabulary |
|---|---|---|---|---|
| `sarvamai/sarvam-30b` (071ae95) | `SarvamMoEForCausalLM` | 18 (19 layers, the first dense) | 128, top-6, 1 shared | 262,144 |
| `gnani/gnani-evon-v3.3-30B-A3B` (fb9ddb6) | `NemotronHForCausalLM` (23 Mamba, 23 MoE, 6 attention layers) | 23 | 128, top-6, 1 shared | 163,072 |
| `Qwen/Qwen3-30B-A3B` (ad44e77), report [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | `Qwen3MoeForCausalLM` | 48 | 128, top-8, none | 151,936 |
| `sarvamai/sarvam-105b` (abe264b), stretch goal | `SarvamMLAForCausalLM` | 31 (32 layers, the first dense) | 128, top-8, 1 shared | 262,144 |

In BF16, a model with about 30B parameters needs roughly 60 GB for weights alone, more than one 48 GB L40S holds, so every replica we add on a 4-GPU setup comes out of KV-cache space.

**Data for traces and traffic mixes**

- FLORES-200, released with NLLB ([arXiv:2207.04672](https://arxiv.org/abs/2207.04672)): parallel sentences across 200 languages, which separates the effect of language from the effect of content.
- IndicTrans2 ([arXiv:2305.16307](https://arxiv.org/abs/2305.16307), TMLR): parallel corpora and benchmarks for all 22 scheduled Indian languages.
- GLUECoS ([arXiv:2004.12376](https://arxiv.org/abs/2004.12376), ACL 2020): English-Hindi code-switched tasks, including language identification.
- L3Cube-HingCorpus ([arXiv:2204.08398](https://arxiv.org/abs/2204.08398)): 52.93M romanised Hinglish sentences from Twitter, with a Hinglish language-identification dataset and model.
- Tamil-English code-mixed corpus ([arXiv:2006.00206](https://arxiv.org/abs/2006.00206), SLTU-CCURL 2020): 15,744 YouTube comments for Tanglish.
- BurstGPT (cluster e) for arrival times and lengths.

## 7. Open scoping questions for the first discussion

1. **Which all-to-all path do we measure?** As we read the code, vLLM's default `allgather_reducescatter` backend moves the same bytes whatever the placement. Do we write a per-GPU deduplicated NCCL all-to-all dispatcher in the plugin, test whether any of the selective backends runs on L40S over PCIe, wait for ThunderEP or CoMoE code, or do all three in sequence? The answer fixes what "all-to-all bytes" means in the paper.
2. **What exactly is the hardware?** Are the 4 and 8 L40S GPUs under one PCIe switch with peer-to-peer enabled, split across two CPU sockets, or host-staged like the consumer cards in ThunderEP and CoMoE? Which provider (E2E Networks or Yotta) and how many hours fit the $250-450 budget? The topology has to be reported, because rank mapping matters.
3. **How do requests reach GPUs?** Placement creates locality only if each GPU's tokens share a language, which means steering requests by detected language to data-parallel attention ranks, as [4] groups micro-batches and Sem-MoE rebatches. How do we keep load balanced when one language dominates, and do we group by family (Indo-Aryan, Dravidian) rather than one language per GPU?
4. **What is the unit of "language"?** ISO language, script or family; and for code-mixed text, romanised Hinglish against Devanagari Hindi and mixed-script Tanglish. OpenMoE's token-ID finding suggests that script may drive early-layer routing as much as language does, and the labels in the atlas must match whatever the steering uses at run time.
5. **Which phase and which latency are we optimising?** Prefill moves many tokens per step, while decode moves few and is memory-bound (METRO, TEMPO). Do we target TTFT, TPOT or end-to-end p99, at which batch sizes and latency targets?
6. **Which models, and what budget?** Sarvam-30B (18 MoE layers, top-6, a shared expert; the brief notes it is deprecated), Gnani Evon v3.3 (a Mamba, attention and MoE hybrid) and Qwen3-30B-A3B (48 MoE layers, top-8, no shared expert). All three have EPLB hooks in both engines' code, but we have not yet run any of them with EP on L40S. Do we keep all three, or two plus a stretch? And what replication budget fits in 48 GB per GPU next to the KV cache?
7. **Which traces and traffic mixes?** The draft exp01 uses Wikipedia articles, whose topics differ by language; FLORES-200 or IndicTrans2's parallel data would separate language from content. Can Sarvam AI or Gnani.ai share an anonymised language mix so that at least one traffic mix is realistic? And confirm the EuroMLSys 2027 deadline and page limit before planning weeks 6 to 9.
