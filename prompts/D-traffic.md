Ablation D: concentrated and shifting traffic.

Read README.md and analyze.py first (analyze.py already accepts --mix). Write ablations/d_traffic.py.

Define six traffic mixes over the ten languages: equal; Hindi-heavy (hi 0.5, en 0.3, the rest shared equally); English-heavy (en 0.6); Indic-only (en 0); and two random mixes drawn from a Dirichlet distribution with concentration 0.5 (seeded). For R = 2 and R = 4 copies per GPU and 4 GPUs, fit the full language-aware plan (groups, layout and copies) on each mix and serve it on every mix, giving a fit-by-serve matrix of traffic cuts on the held-out half. For every serve mix, also compute the language-blind policy of Ablation A at the same R.

Save results/ablations/D/table.csv, heatmaps results/ablations/D/fit_by_serve_R2.png and fit_by_serve_R4.png, and results/ablations/D/NOTES.md answering two questions with numbers: does language awareness add more over the language-blind policy when traffic is concentrated, and how much traffic cut is lost when the plan was fitted on a different mix.

Keep all randomness seeded, do not modify analyze.py or anything in results/exp01/, and finish by printing the main table and three bullet points with the key numbers.
