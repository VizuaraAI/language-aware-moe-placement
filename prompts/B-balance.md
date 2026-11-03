Ablation B: balance-aware placement. This is the method contribution of the paper.

Read README.md, analyze.py and results/ablations/A/NOTES.md first. Write ablations/b_balance.py.

For each layer, place the experts on the GPUs (each GPU holds exactly experts/GPUs of them) so as to maximise the traffic-weighted share of hits on each language's home GPU, subject to every GPU's expert load staying at or below cap times the mean load, where load is measured on the fit half. Solve it exactly as an integer program with PuLP and its bundled CBC solver (binary variable per expert and GPU), and also implement a fast greedy heuristic; compare the two on objective value and solve time.

Sweep cap in {1.1, 1.25, 1.5, 2.0, no cap} and replica budgets R in {0, 2, 4}, where copies are added after placement as in policy 3 of Ablation A. Evaluate every plan on the held-out half: local share, traffic cut against the default layout, and imbalance. Use the language groups from analyze.py for 4 GPUs, and repeat for 8 GPUs if the exact solver finishes in reasonable time.

Save results/ablations/B/table.csv, a figure results/ablations/B/frontier.png with traffic cut against imbalance (one point per plan, with the default layout and the language-blind policy of Ablation A at the same R as reference points), and results/ablations/B/NOTES.md stating which cap gives the best trade-off and how close the greedy heuristic comes to the exact solution.

Keep all randomness seeded, do not modify analyze.py or anything in results/exp01/, and finish by printing the main table and three bullet points with the key numbers.
