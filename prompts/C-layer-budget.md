Ablation C: how to spend a fixed memory budget across layers.

Read README.md and analyze.py first. Write ablations/c_layer_budget.py.

Give each GPU a total budget equal to 2 extra experts per layer on average across all 48 layers (96 copies per GPU in total). Compare four ways to spend it, all on top of the language-aware layout from analyze.py with copies chosen by each GPU's own languages:
1. Uniform: 2 copies in every layer.
2. By specificity: copies in proportion to each layer's language specificity (the Jensen-Shannon measure in analyze.py), rounded so the total is exact.
3. Greedy by gain: repeatedly give the next copy to the layer and GPU where it saves the most cross-GPU traffic on the fit half.
4. Concentrated: the whole budget in the 12 most language-specific layers, then in the 24 most specific.

Evaluate each on the held-out half for 4 and 8 GPUs: total traffic cut against the default layout, imbalance, and the traffic saved per layer. Save results/ablations/C/table.csv, results/ablations/C/per_layer.png (traffic saved per layer for each allocation) and results/ablations/C/NOTES.md stating which allocation wins and by how much.

Keep all randomness seeded, do not modify analyze.py or anything in results/exp01/, and finish by printing the main table and three bullet points with the key numbers.
