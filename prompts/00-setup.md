Read README.md and analyze.py first.

Create a Python virtual environment in .venv, install requirements.txt, and run `python analyze.py`. Then open results/summary.json and check two numbers for 4 GPUs: local_share_aware should be 0.344 and local_share_baseline 0.257 (within 0.001). Tell me whether they match.

Then explain to me, in five plain sentences, what analyze.py simulates: what a "home GPU", the "local share", "replicas" and the "imbalance" mean, and which half of the text is used to fit a placement and which half to evaluate it. Do not change analyze.py.
