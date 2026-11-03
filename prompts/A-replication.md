Ablation A: the replication budget, against a language-blind baseline.

Read README.md and analyze.py first. Write a new script, ablations/a_replication.py, that loads results/routing_qwen3_30b_a3b.npz the same way analyze.py does and reuses its functions (profits, place_layer, balanced_groupings, simulate, js_bits) where they fit. Fit every placement on the first half of the text and evaluate it on the second half, exactly as analyze.py does, with equal traffic per language.

For 4 and 8 GPUs and replica budgets R in {0, 1, 2, 4, 8, 16} extra experts per GPU per layer, compare three policies at the same R:
1. Language-blind: the default layout (expert e on GPU e // experts_per_gpu), each language's tokens spread evenly over all GPUs, and each GPU also holding copies of the R globally busiest experts it does not already have.
2. Language-aware layout with language-blind copies: the language groups, homes and layout from analyze.py, with each GPU copying the R globally busiest experts it does not have.
3. Fully language-aware: the same layout, with each GPU copying the R experts its own languages use most.

For each policy, GPU count and R, report: the local share (hits on the token's own GPU), the cross-GPU traffic cut against the default layout with no copies, the expert-load imbalance (busiest GPU over the mean, with copies serving local requests) and the memory overhead (R divided by experts per GPU).

Save results/ablations/A/table.csv, a figure results/ablations/A/frontier.png with traffic cut on the y-axis and memory overhead on the x-axis (one panel per GPU count, one line per policy), and results/ablations/A/NOTES.md stating, with numbers, at which budgets language awareness adds most over the language-blind policy and where the gap closes.

Keep all randomness seeded, do not modify analyze.py or anything in results/exp01/, and finish by printing the main table and three bullet points with the key numbers.
