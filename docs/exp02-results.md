# Exp02: language structure in a second model and on parallel text

Run on 5 November 2026 by the Vizuara research team.

## Summary

The English against Indic split in Exp01 is not a Wikipedia artefact and is not peculiar to Qwen3. It appears in Sarvam-30B, and it holds on parallel text, at 92% of its Wikipedia size on Qwen3 and 77% on Sarvam. At 2 copies per GPU on 4 GPUs, language awareness adds about 10 to 11 percentage points of traffic cut over language-blind copies on both models and both text sets, well above the 2 points in the rule.

Three results limit that claim. Differences among the Indian languages roughly halve on parallel text. Our prediction for code-mixed text failed on both models. One pre-registered control, the padding check, also failed on both models, so FLORES ran one sentence at a time, as the criteria written before that diagnostic required.

The plan had three models and we ran two. Gnani Evon v3.3 was skipped because of its licence, as explained below.

## What we ran

We ran Qwen3-30B-A3B (48 MoE layers, top-8) and Sarvam-30B (18 MoE layers, top-6, plus a shared expert that is not counted) in BF16 on one H100. For every token at every MoE layer we recorded the experts the model used. Counts are stored per sequence, and the text itself is not kept.

| Text | Source | Qwen3 tokens | Sarvam tokens |
|-|-|-|-|
| Wikipedia, ten languages | 96 sequences of 512 tokens each | 49,152 English bridge; the rest from Exp01 | 491,520 |
| FLORES, ten languages | Same 500 sentences in each language, one at a time | 729,791 | 184,414 |
| Hinglish and its English side | cmu_hinglish_dog, 96 sequences per side | 98,304 | 98,304 |
| Tanglish | offenseval_dravidian, 28,338 of 35,139 comments kept by a Latin-script filter | 49,152 | 49,152 |

FLORES-101, Goyal et al., CC BY-SA 4.0; used for evaluation, not redistributed.

The decision rule was hashed at 00:23 IST, before any smoke test or routing run (SHA-256 bb4e6eca, in `logs/decision_rule_sha256_at_write.log`). The file has not changed since, and `analyze.py` refuses to run if it does. Nobody outside the run has reviewed the rule. The code did change after the hash, and each change is recorded in `logs/code_changes_after_hash.log`.

## Main results

D is the mean Jensen-Shannon divergence in bits between English and each Indic language, and the floor is the same measure between two halves of one language.

| Measure | Qwen3-30B-A3B | Sarvam-30B |
|-|-|-|
| Layers with D at least 3 times the floor, Wikipedia | 48 of 48 | 18 of 18 |
| D on Wikipedia (floor) | 0.318 (0.013) | 0.274 (0.018) |
| D on FLORES (floor) | 0.294 (0.010) | 0.212 (0.015) |
| FLORES over Wikipedia, literal ratio | 0.923 | 0.774 |
| FLORES over Wikipedia, equal-sample ratio | 0.927 | 0.768 |
| Between Indic languages, Wikipedia | 0.035 | 0.071 |
| Between Indic languages, FLORES | 0.017 | 0.038 |
| Hinglish position, early layers (0 English, 1 Hindi) | 0.657 | 0.616 |
| Hinglish position, late layers | 0.713 | 0.574 |
| Gap at 2 copies, 4 GPUs, Wikipedia (points) | 10.06 | 11.18 |
| Gap at 2 copies, 4 GPUs, FLORES (points) | 9.67 | 10.64 |

D by band, in bits:

| Band | Qwen3 Wikipedia | Qwen3 FLORES | Sarvam Wikipedia | Sarvam FLORES |
|-|-|-|-|-|
| Early | 0.333 | 0.318 | 0.184 | 0.128 |
| Middle | 0.257 | 0.221 | 0.223 | 0.161 |
| Late | 0.548 | 0.560 | 0.567 | 0.500 |

The three placement policies at 2 copies per GPU on 4 GPUs (traffic cut in percent, then busiest GPU over the mean):

| Model and text | Language-blind | Aware layout, blind copies | Fully aware |
|-|-|-|-|
| Qwen3, Wikipedia | 22.36, 1.25 | 32.11, 1.98 | 32.42, 1.99 |
| Qwen3, FLORES | 21.94, 1.25 | 31.43, 2.20 | 31.61, 2.18 |
| Sarvam, Wikipedia | 17.76, 1.23 | 28.19, 1.52 | 28.94, 1.51 |
| Sarvam, FLORES | 16.68, 1.20 | 26.72, 1.59 | 27.32, 1.58 |

All budgets, and 8 GPUs, are in `summary.json`.

## Rule outcomes

**G0, sound measurement: passed except for padding.** The Qwen3 English bridge matches Exp01's counts in every cell, with 0 of 18,874,368 expert hits moved, under a new hash-locked environment. On both models the router's indices equal a hand computation for 100% of tokens, and a separate count at the experts themselves matches the gate exactly. The sentinel batch was identical at the start and end of each run. The label permutation null gave 0.0012 bits on Qwen3 and 0.0018 on Sarvam, against real divergences of 0.240 and 0.184. Totals were exact in every set. Negative log likelihood was 1.60 and 0.67 on Qwen3 and 2.26 and 2.20 on Sarvam (English, then Hindi), so the models loaded correctly. The padding control failed, and the next section covers it.

**G1, does language structure exist: passed on both models.**

**G2, does it survive parallel text: survives on both models.** Both ratios are above 0.5. Sentence length does not explain Sarvam's fall: measured against only the first 64 tokens of each Wikipedia sequence, the ratio is 0.801.

**G3, code-mixed prediction: not supported.** We predicted that romanised Hinglish would sit closer to English than to Hindi in early layers and move toward Hindi later. In early layers it sits closer to Hindi on both models, with 95% intervals of 0.647 to 0.666 on Qwen3 and 0.606 to 0.624 on Sarvam. It then moves toward Hindi on Qwen3, as predicted (late minus early, 0.052 to 0.062), and toward English on Sarvam, against the prediction (minus 0.049 to minus 0.033).

This reading depends on the reference. With Wikipedia as the reference the early result is the same (0.625 and 0.640). With the Hinglish set's own English side and Wikipedia Hindi, the position is 0.490 and 0.457 in early layers and 0.367 and 0.340 in middle layers, so Hinglish is then closer to English. That reference shares content and register with the Hinglish, while the Hindi does not, so it leans toward English.

Register alone moves routing by about as much as language does in early and middle layers. English chat against FLORES English is 0.187, 0.221 and 0.165 bits on Qwen3 (early, middle, late), against 0.204, 0.135 and 0.411 for Hinglish against its English side. Sarvam shows the same pattern, with 0.152, 0.236 and 0.159 against 0.127, 0.135 and 0.380. Language separates the two most clearly in late layers.

Tanglish sits closer to Tamil than to English in every band on both models (0.615, 0.672 and 0.632 on Qwen3; 0.592, 0.582 and 0.517 on Sarvam). Its English control is weak, and at least some comments are not Tamil.

**Outcome for the paper: the narrowing condition is not met.** The rule needs a gap below 2 points on both new models, and Sarvam's is 11.18 on Wikipedia and 10.64 on FLORES. Gnani cannot change this. The gap shrinks as the budget grows, from 13.50 points with no copies to 5.43 with 16 copies on Sarvam on Wikipedia, and from 12.68 to 4.78 on Qwen3.

## The padding control failed

The rule required at least 99% agreement between sentences padded in a batch and the same sentences alone. Agreement was 88.1% and 82.2% on Qwen3 and 83.8% and 85.0% on Sarvam (English, then Hindi). A batch of 8 with no padding at all agreed with a sequence alone for only 90.7% and 89.3% of token-layer pairs.

Before running a diagnostic we wrote down what would explain this (`logs/padding_diagnostic_criteria.log`). Two tests changed tokens without changing the batch shape: the tokens after a sentence's end, and the other rows of the batch. We required 100% identical routing, and got 54.1% and 66.8% on Qwen3. That test had two faults, both ours. It compared ordered indices, not sets, and it assumed that rounding would stay fixed when the shape did, although batch mates change the size of each expert's sub-batch. Measured as sets, agreement is 84.8% and 91.0% on Qwen3 and 86.2% and 90.3% on Sarvam, which still falls short of 100%.

The criteria said that FLORES must then run one sentence at a time, and it did, on both models. That change is the main reason the GPU time rose above plan.

The likely cause is BF16 rounding that moves experts across the top-k boundary, followed by a cascade through later layers. The evidence fits but does not prove it. The median change in the hidden state is 0.0 at the first MoE layer and reaches between 2.9% and 8.2% at the last. On Qwen3, flipped tokens had a median relative margin of 0.012 at the boundary, against 0.061 for unflipped ones, and 7.6% of tokens had an exact tie there. Most hits stayed put: at least 97.6% are kept in these comparisons. On the counts themselves, batching against single sentences moved 0.23% and 0.15% of hits on Qwen3 FLORES, and in every layer the resulting divergence stayed below half of the floor. We did not run that count comparison on Sarvam's FLORES.

Wikipedia, Hinglish and Tanglish remain in batches of 8 with no padding, because the bridge to Exp01 needs them that way. Batching 8 Wikipedia sequences against running them alone moved 0.32% of hits on Qwen3 and 0.46% on Sarvam. We did not measure this as a divergence, so that part of the explanation rests on the share of hits moved.

This matters beyond the control. The expert set chosen for a token depends on the batch it travels with for about 10% of token-layer pairs, so a trace recorded at one batch shape is not an exact record of routing at another.

## What it means for the paper

The first claim gets stronger. The literature review says Exp01's middle-layer divergence may be inflated by Wikipedia's topics, and parallel text puts a size on that: FLORES keeps 86% of it on Qwen3 and 72% on Sarvam. The review can now say that Exp01's effect is reduced, and not removed, by controlling content.

The claim should be about English against Indic. Between Indic languages the divergence on FLORES is only about 1.7 times the floor on Qwen3 and 2.5 times on Sarvam, and it is half what Wikipedia showed. These two sets differ in more than content, so this is a reason for caution and not a measurement of content alone. Ablation D, which tests Indic-only traffic, is now the more important test of whether language-aware placement helps within India.

The gain comes mainly from the layout and from sending each language to a home GPU, not from choosing copies by language. At 2 copies, language-blind copies on an aware layout come within 0.2 to 0.8 points of fully aware copies. Tokens also have to be routed by language for this to work. The price is balance: 1.51 against 1.23 on Sarvam and 1.99 against 1.25 on Qwen3 on Wikipedia. The gap is therefore not measured at equal balance, and the Ablation B planner still has to show how much of it survives a load cap.

Code-mixed text should not be described as behaving like English. The paper needs a Hindi chat reference in Devanagari before it says where Hinglish falls, because no reference here matches Hinglish in register. Until then it can be treated as its own traffic class.

Anyone porting Exp01's code to Sarvam must read the gate's indices. A top-k of its raw logits matches the routed set for only 18.4% of tokens, against 100% on Qwen3.

Sarvam's tokenizer averages 2.7 to 3.9 characters per token on Indic Wikipedia, against 4.6 on English. Qwen3's averaged 0.67 to 1.08 on Indic FLORES. Sarvam's early-layer divergence is also smaller, 0.184 against 0.333 bits. The tokenizer may therefore account for part of Qwen3's early split, but we did not test this.

## What differs from the plan

- **Gnani Evon v3.3 was not run.** It is not gated and is tagged Apache 2.0. However, its NOTICE keeps the underlying NVIDIA work under the NVIDIA Nemotron Open Model License, which says that using the work means agreeing to be bound by it, and which includes governing-law and venue terms. The brief says to skip a model whose licence requires acceptance. Its weights were not downloaded. The plan names no fallback, and we did not substitute one. If the licence is accepted later, the same rule applies and the run would be reported as a later addition. Its adapter is not in this code.
- **FLORES ran one sentence at a time**, for the reason above.
- **Co-occurrence counts were not recorded.** Exp01 stored them, and no gate here uses them.
- **Qwen3 used 32 GiB of memory, not 64,** after a 24 GiB peak in the smoke test. Sarvam's smoke checks and main run shared one container to avoid a second load.
- **Sarvam's tokenizer is built directly from `tokenizer.json`.** `AutoTokenizer` fails in transformers 4.57.2 on a local folder with this config, because of a bug in a Mistral patch.

## Other caveats

- This is routing with no serving engine. Traffic numbers are a simulation with equal traffic per language and random search over 200 language groupings, and no latency was measured.
- FLORES is short, news-like text, unlike chat. Wikipedia topics differ by language. We ran BF16, and Sarvam's float32 weights are converted on load.
- The Hinglish and English sides are the same conversations in the same order, but they reach 49,152 tokens after different numbers of turns: 2,517 and 3,640 on Qwen3, 2,915 and 3,480 on Sarvam.
- Equal-sample figures thin expert hits independently, although hits from one token are correlated. The bootstrap treats sentences and sequences as independent and covers sampling only, not the choice of reference, so those narrow intervals should not be read as covering the sensitivity above.
- The Qwen3 container restarted once after 62 minutes, probably a preemption. Finished languages were kept and the partial one was rerun. Its start and end sentinel covers the last container only.
- The added Sarvam diagnostics are descriptive and were added after we saw the Qwen3 failure.

## Cost

Modal billed $9.51 against a plan of about $8.60 for three models, with 2.00 H100 hours against a planned 1.65. Our own log estimated $9.11. The largest bills were the Qwen3 run ($6.53), the Sarvam run ($1.33) and the padding diagnostic ($0.94). A failed first Qwen3 smoke test cost $0.16, and two failed Sarvam starts cost about $0.01 each. Volume storage is not in this figure. Sarvam's weights and all text were deleted from the volume `irb08-exp02` after the results were copied here. Exp01's volume was mounted read only, and no apps are left running.

## Files

- `exp02_modal.py`, `ctl.py`, `lock_modal.py`, `analyze.py`: the code. `exp01_analyze.py` is Exp01's `analyze.py`, byte for byte, which the self test checks.
- `requirements/`: inputs and hash locks for both environments.
- `decision_rule.json`, `logs/decision_rule_sha256_at_write.log`: the rule and its hash.
- `summary.json`: every number above, including per-layer values and all budgets.
- `results/out/`: routing counts per sequence and control outputs. `results/text_manifest.json` holds text hashes, `results/costlog.jsonl` our cost log, and `results/billing.json` Modal's billing rows.
- `logs/`: raw logs of each step, the diagnostic criteria and the code changes after the hash.
