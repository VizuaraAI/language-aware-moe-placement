# Exp01: which experts each language uses in Qwen3-30B-A3B

Run by the Vizuara research team in the first week of the project.

## What we ran

Qwen3-30B-A3B (48 layers, 128 experts, 8 chosen per token) in bf16 on one H100, with Hugging Face transformers. The text
is Wikipedia (the 20231101 dump) in ten languages: English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada,
Malayalam and Punjabi, 96 sequences of 512 tokens per language (49,152 tokens each). For every token at every layer we
recorded the eight experts the router chose. The first half of each language's text is used to design a placement and
the second half to score it, so no number below is measured on the text that produced it. The run took 8.4 minutes of
GPU time, about one dollar.

## What we found

1. **English routes differently from the Indian languages at every layer.** The Jensen-Shannon divergence between
   English and the Indian languages averages 0.33 bits in layers 1-8, 0.26 bits in layers 9-40 and 0.55 bits in layers
   41-48. Between two Indian languages it is about 0.04 bits, against a sampling noise floor of 0.013 bits. The language
   effect is real, but it is mostly an English-against-Indic split.
2. **A layout alone buys little.** In a simulation of expert parallelism with equal traffic per language, a
   language-aware layout raises the share of expert hits that stay on the token's own GPU from 25.7% to 34.4% on 4 GPUs
   (12.7% to 22.6% on 8 GPUs). That cuts cross-GPU traffic by about 12%.
3. **A small replication budget does most of the work.** Copying the four experts each GPU's languages use most onto
   that GPU, per layer, raises the local share to 59.6% on 4 GPUs and 51.1% on 8 GPUs: about 46% and 44% less cross-GPU
   traffic.
4. **The price is balance.** A layout built only for locality piles busy experts onto some GPUs. The busiest GPU
   carries 2.32 times the mean expert load on 4 GPUs (the default layout: 1.40), and 3.77 times on 8 GPUs (default:
   1.77). When replicas serve local requests the imbalance falls to 1.75 and 2.67.
5. **The tokenizer taxes the Indian languages.** Qwen3's tokenizer averages 4.56 characters per token in English and
   0.69 to 1.10 in the nine Indian languages.

## What it means for the paper

The question becomes how to trade locality against balance: a planner that places experts and spends a small
replication budget under a load cap, layer by layer. The serving measurement must then show whether about 45% less
all-to-all traffic turns into lower p99 latency over PCIe. If the gains among Indian languages stay this small, the
paper should say so plainly: the English-against-Indic split carries most of the effect.

## Caveats

This is a simulation of all-to-all volume, not a serving measurement; it covers one model, Wikipedia text and equal
traffic per language; the language-to-GPU groups come from a random search over 300 balanced groupings; load imbalance
counts expert work only, not attention.

## Files

`exp01_routing.py` (the Modal run), `analyze.py` (analysis and ablations; runs on a laptop), `results/` (routing counts,
`summary.json`, figures 1-4).
