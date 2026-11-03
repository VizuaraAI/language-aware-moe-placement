"""Analysis for exp01 (topic 08): language specificity per layer, and a placement simulation for expert parallelism.

Input:  results/routing_qwen3_30b_a3b.npz (from exp01_routing.py).
Output: results/summary.json, results/fig1_specificity.png, results/fig2_local_share.png, results/fig3_imbalance.png.

Model of the system (a simulation, not a serving measurement): with expert parallelism over G GPUs, each GPU holds
E/G experts per layer. Requests of each language are served on a "home" GPU, and every (token, expert) pair whose
expert sits on another GPU costs all-to-all traffic. With the default layout and language-blind serving, about 1/G of
the pairs are local. A language-aware layout groups languages onto home GPUs and, layer by layer, places on each GPU
the experts its languages use most (an exact assignment problem). It is fitted on the first half of the text and
scored on the second half, so the gain is not measured on the data that produced it.
"""
import json
import pathlib
import random

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results"
SEED = 0


def js_bits(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.maximum(b, 1e-12)), 0.0), axis=-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def place_layer(prof, slots):
    """prof[e, g] = traffic-weighted use of expert e by the languages homed on GPU g. Exact balanced assignment."""
    cost = -np.repeat(prof, slots, axis=1)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(prof.shape[0], dtype=int)
    out[rows] = cols // slots
    return out


def profits(p, groups, w, layer):
    n_exp = p.shape[-1]
    return np.stack([sum((w[l] * p[l, layer] for l in g), np.zeros(n_exp)) for g in groups], axis=1)


def simulate(p_fit, p_ev, groups, w, replicas=0):
    n_lang, n_layers, n_exp = p_fit.shape
    gpus = len(groups)
    slots = n_exp // gpus
    home = np.empty(n_lang, dtype=int)
    for g, ls in enumerate(groups):
        home[ls] = g
    local, local_rep, base, imb_aware, imb_base, imb_rep = (np.zeros(n_layers) for _ in range(6))
    default = np.arange(n_exp) // slots
    for layer in range(n_layers):
        prof = profits(p_fit, groups, w, layer)
        where = place_layer(prof, slots)
        # replicas: each GPU also keeps copies of the `replicas` remote experts its own languages use most
        extra = np.zeros((gpus, n_exp), dtype=bool)
        if replicas:
            for g in range(gpus):
                cand = np.where(where != g)[0]
                extra[g, cand[np.argsort(-prof[cand, g])[:replicas]]] = True
        for l in range(n_lang):
            g = home[l]
            here = where == g
            local[layer] += w[l] * p_ev[l, layer, here].sum()
            local_rep[layer] += w[l] * p_ev[l, layer, here | extra[g]].sum()
            base[layer] += w[l] * p_ev[l, layer, default == g].sum()
        load = (w[:, None] * p_ev[:, layer]).sum(0)  # expert compute load, all languages
        per_gpu = np.array([load[where == g].sum() for g in range(gpus)])
        per_gpu_base = np.array([load[default == g].sum() for g in range(gpus)])
        imb_aware[layer] = per_gpu.max() / per_gpu.mean()
        # with replicas, a GPU serves its own languages' hits on replicated experts locally; the rest go to the primary copy
        rep_load = np.zeros(gpus)
        for l in range(n_lang):
            g = home[l]
            for e in range(n_exp):
                served = g if (where[e] == g or extra[g, e]) else where[e]
                rep_load[served] += w[l] * p_ev[l, layer, e]
        imb_rep[layer] = rep_load.max() / rep_load.mean()
        imb_base[layer] = per_gpu_base.max() / per_gpu_base.mean()
    total = w.sum()
    return {"local": local / total, "local_rep": local_rep / total, "base": base / total,
            "imb_aware": imb_aware, "imb_base": imb_base, "imb_rep": imb_rep,
            "token_imbalance": float(max(w[g].sum() for g in groups) / (total / gpus))}


def balanced_groupings(n_lang, gpus, samples, rng):
    sizes = [n_lang // gpus + (1 if i < n_lang % gpus else 0) for i in range(gpus)]
    for _ in range(samples):
        order = list(range(n_lang))
        rng.shuffle(order)
        groups, i = [], 0
        for s in sizes:
            groups.append(order[i:i + s])
            i += s
        yield groups


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Exp01 analysis and ablations (runs on a laptop, no GPU).")
    ap.add_argument("--mix", default="", help='traffic weights, e.g. "hi=0.5,en=0.3" (unlisted languages share the rest equally)')
    ap.add_argument("--replicas", type=int, default=4, help="replicated experts per GPU per layer")
    ap.add_argument("--gpus", default="4,8", help="GPU counts to simulate, e.g. 2,4,8")
    ap.add_argument("--tag", default="", help="suffix for the output files of an ablation run")
    return ap.parse_args()


def weights(langs, mix):
    w = np.ones(len(langs)) / len(langs)
    if mix:
        given = {k: float(v) for k, v in (x.split("=") for x in mix.split(","))}
        rest = [l for l in langs if l not in given]
        left = max(0.0, 1.0 - sum(given.values()))
        w = np.array([given.get(l, left / max(1, len(rest))) for l in langs])
    return w / w.sum()


def main():
    args = parse_args()
    d = np.load(RES / "routing_qwen3_30b_a3b.npz")
    counts, meta = d["counts"].astype(float), json.loads(str(d["meta"]))
    langs = list(meta["langs"])
    n_layers = meta["layers"]
    p_fit = counts[0] / counts[0].sum(-1, keepdims=True)
    p_ev = counts[1] / counts[1].sum(-1, keepdims=True)
    w = weights(langs, args.mix)  # equal traffic per language unless --mix says otherwise
    gpu_list = [int(g) for g in args.gpus.split(",")]
    tag = f"_{args.tag}" if args.tag else ""

    mix = p_fit.mean(0)
    spec = js_bits(p_fit, mix[None])     # [lang, layer]
    floor = js_bits(p_fit, p_ev)         # same language, other half: sampling noise
    rng = random.Random(SEED)

    results = {"model": meta["model"], "langs": meta["langs"], "tokens": meta["tokens"],
               "chars_per_token": meta["chars_per_token"], "layers": n_layers, "experts": meta["experts"],
               "top_k": meta["top_k"], "specificity_bits_by_layer": spec.mean(0).round(4).tolist(),
               "noise_floor_bits_by_layer": floor.mean(0).round(4).tolist(), "ep": {}}
    bands = {"early (layers 1-8)": slice(0, 8), "middle (9-40)": slice(8, n_layers - 8),
             f"late ({n_layers - 7}-{n_layers})": slice(n_layers - 8, n_layers)}
    sims = {}
    for gpus in gpu_list:
        best, best_score = None, -1.0
        for groups in balanced_groupings(len(langs), gpus, 300, rng):
            score = simulate(p_fit, p_fit, groups, w)["local"].mean()  # choose on the fit half only
            if score > best_score:
                best, best_score = groups, score
        sim = simulate(p_fit, p_ev, best, w, replicas=args.replicas)
        sims[gpus] = sim
        rem_aware, rem_base = 1 - sim["local"], 1 - sim["base"]
        results["ep"][gpus] = {
            "groups": [[langs[i] for i in g] for g in best],
            "local_share_baseline": round(float(sim["base"].mean()), 4),
            "local_share_aware": round(float(sim["local"].mean()), 4),
            "replicas_per_gpu": args.replicas,
            "local_share_aware_plus_replicas": round(float(sim["local_rep"].mean()), 4),
            "remote_traffic_cut_pct": round(float(100 * (1 - rem_aware.mean() / rem_base.mean())), 1),
            "remote_traffic_cut_pct_with_replicas": round(float(100 * (1 - (1 - sim["local_rep"]).mean() / rem_base.mean())), 1),
            "by_band": {name: {"baseline": round(float(sim["base"][s].mean()), 4),
                               "aware": round(float(sim["local"][s].mean()), 4),
                               "aware_plus_replicas": round(float(sim["local_rep"][s].mean()), 4)} for name, s in bands.items()},
            "expert_load_imbalance_aware_mean": round(float(sim["imb_aware"].mean()), 3),
            "expert_load_imbalance_aware_max": round(float(sim["imb_aware"].max()), 3),
            "expert_load_imbalance_default_mean": round(float(sim["imb_base"].mean()), 3),
            "expert_load_imbalance_with_replicas_mean": round(float(sim["imb_rep"].mean()), 3),
            "token_load_imbalance": round(sim["token_imbalance"], 3),
        }
    results["traffic_weights"] = {l: round(float(x), 4) for l, x in zip(langs, w)}
    # pairwise distance between languages (JS, bits), averaged over early, middle and late layers
    pair = np.array([[js_bits(p_fit[a], p_fit[b]) for b in range(len(langs))] for a in range(len(langs))])  # [a, b, layer]
    results["pairwise_js_bits"] = {name: pair[:, :, s].mean(-1).round(4).tolist() for name, s in bands.items()}
    (RES / f"summary{tag}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    x = np.arange(1, n_layers + 1)
    names = [meta["langs"][k] for k in langs]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(np.maximum(spec - floor, 0), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(langs)), names)
    ax.set_xticks(range(0, n_layers, 4), [str(i + 1) for i in range(0, n_layers, 4)])
    ax.set_xlabel("Layer")
    ax.set_title(f"How far each language's expert use departs from the average ({meta['model']}), noise floor removed")
    fig.colorbar(im, ax=ax, label="JS divergence (bits)")
    fig.tight_layout()
    fig.savefig(RES / f"fig1_specificity{tag}.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    vmax = max(float(pair[:, :, s].mean(-1).max()) for s in bands.values())
    for ax, (name, s) in zip(axes, bands.items()):
        im = ax.imshow(pair[:, :, s].mean(-1), cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(langs)), names, rotation=60, fontsize=8)
        ax.set_yticks(range(len(langs)), names, fontsize=8)
        ax.set_title(name[0].upper() + name[1:], fontsize=10)
    fig.suptitle(f"Routing distance between languages ({meta['model']}, Jensen-Shannon divergence in bits)", fontsize=11)
    fig.colorbar(im, ax=axes, label="JS divergence (bits)", shrink=.8)
    fig.savefig(RES / f"fig4_pairwise{tag}.png", dpi=150, bbox_inches="tight")

    fig, axes = plt.subplots(1, len(gpu_list[:2]), figsize=(12, 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, gpus in zip(axes, gpu_list[:2]):
        s = sims[gpus]
        ax.plot(x, s["local"], label="language-aware layout")
        ax.plot(x, s["local_rep"], label=f"language-aware + {args.replicas} replicas per GPU")
        ax.plot(x, s["base"], label="default layout", color="grey")
        ax.set_title(f"{gpus} GPUs: share of expert hits on the token's own GPU")
        ax.set_xlabel("Layer")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("Local share (held-out text)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / f"fig2_local_share{tag}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for gpus in gpu_list[:2]:
        ax.plot(x, sims[gpus]["imb_aware"], label=f"{gpus} GPUs, language-aware")
        ax.plot(x, sims[gpus]["imb_rep"], linestyle=":", label=f"{gpus} GPUs, language-aware + replicas")
        ax.plot(x, sims[gpus]["imb_base"], linestyle="--", label=f"{gpus} GPUs, default")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert load, max / mean GPU")
    ax.set_title("Load imbalance that the layout creates")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / f"fig3_imbalance{tag}.png", dpi=150)
    print(json.dumps({k: results["ep"][k] for k in results["ep"]}, indent=1))


if __name__ == "__main__":
    main()
