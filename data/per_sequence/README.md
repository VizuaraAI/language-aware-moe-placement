# Per-sequence routing counts

These files hold the expert routing counts behind Exp01 and Exp02, one row per sequence, so that a shuffled-label control can be run exactly. For every sequence they record which experts each MoE layer chose and how often. No text is included: no sentences, no prompts and no token IDs, only counts, labels and token counts.

## Files

| File | What it holds |
|-|-|
| `qwen3_per_sequence.npz` | Qwen3-30B-A3B, 5,384 sequences. `counts` has shape (5384, 48, 128). |
| `qwen3_index.csv` | One line per row of `qwen3_per_sequence.npz`, with the same labels. |
| `sarvam_per_sequence.npz` | Sarvam-30B, 6,248 sequences. `counts` has shape (6248, 18, 128). |
| `sarvam_index.csv` | One line per row of `sarvam_per_sequence.npz`. |
| `qwen3_wikipedia_exp01_split_totals.npz` | Exp01's Qwen3 Wikipedia counts for all ten languages, summed over the fit half and the evaluation half. Not per sequence; see the gap below. |

`counts[i, layer, expert]` is the number of tokens in sequence `i` that the router sent to `expert` at that MoE layer. Each token is sent to k experts per layer (8 on Qwen3, 6 on Sarvam), so for every sequence and layer the counts sum to k times `n_tokens`. We checked this for every row. The arrays are `uint16`.

Each per-sequence `.npz` also stores these label arrays, aligned with the first axis of `counts`, and the CSV repeats them with a `row` column:

| Field | Meaning |
|-|-|
| `seq_id` | `{subset}-{language}-{position}`, for example `flores-hi-0017`. Unique within a model. |
| `source` | `wikipedia`, `flores` or `code_mixed`. |
| `subset` | `wiki`, `flores`, `hinglish` or `tanglish`. |
| `language` | `en`, `hi`, `bn`, `ta`, `te`, `mr`, `gu`, `kn`, `ml`, `pa`; `hi_en` for romanised Hinglish; `ta_en` for romanised Tanglish. |
| `split` | `fit`, `eval` or `none` (see below). |
| `n_tokens` | Tokens in the sequence, counted by that model's own tokenizer. |
| `position_in_set` | Position of the sequence within its subset and language, starting at 0. |
| `flores_parallel_index` | For FLORES, the sentence's position in the FLORES-101 dev split, starting at 0. The same index is the same sentence in every language. `-1` elsewhere. |
| `dataset` | The Hugging Face dataset the text came from (CSV only). |

Loading one model takes a few lines:

```python
import numpy as np
d = np.load("sarvam_per_sequence.npz")
counts, lang, split = d["counts"], d["language"], d["split"]
fit_hi = counts[(d["source"] == "flores") & (lang == "hi") & (split == "fit")].sum(0)   # (layers, experts)
```

## Models

| Model | Revision | MoE layers | Routed experts | k |
|-|-|-|-|-|
| `Qwen/Qwen3-30B-A3B` | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | 48 (every decoder layer) | 128 | 8 |
| `sarvamai/sarvam-30b` | `071ae95e933605ca1104a6b4524a36a98488efa4` | 18 | 128 | 6 |

Both ran in BF16 on one H100 with no serving engine. On Qwen3 the routed set is the top 8 of the gate's logits, which matched a hand computation for every token. On Sarvam it is the set of indices the gate itself returns. A top-k of Sarvam's raw logits matches that set for only 18.4% of tokens, so anyone recomputing routing on Sarvam should read the gate's indices. Sarvam's shared expert is not counted, and its layer axis indexes its 18 MoE layers in order and leaves out any layer without a router.

## Splits

The splits are the ones Exp01 and Exp02 used, decided by position and nothing else.

- **Wikipedia:** 96 sequences per language. Positions 0 to 47 are `fit` and 48 to 95 are `eval`.
- **FLORES:** 500 sentences per language. Positions 0 to 249 are `fit` and 250 to 499 are `eval`. Because FLORES is parallel, the fit half holds the same 250 sentences in every language.
- **Code-mixed:** `none`. These sets did not enter the fit and evaluation design. Exp02 compared their first 48 and last 48 sequences only to estimate a split-half noise floor.

Placement was fitted on `fit` and scored on `eval`. The noise floor in both experiments is the divergence between a language's fit half and its own evaluation half.

## How the sequences were made

**Wikipedia.** From the `20231101` dump of `wikimedia/wikipedia` (revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`), we took articles in dataset order, kept those of at least 1,500 characters and 512 tokens, and cut each to its first 512 tokens until there were 96. Each sequence is one article. The 512-token test uses each model's tokenizer, so position 17 need not be the same article on both models.

**FLORES.** The first 500 sentences of the FLORES-101 dev split (`gsarti/flores_101`, revision `1a45e707ea4b5ea3d6c71341f18bfca6a6e356a3`), one sentence per sequence, each run alone with no padding.

**Code-mixed.** Hinglish comes from `festvox/cmu_hinglish_dog` (revision `19796c6fb32020154cb2745d48704fa73e29b17d`). Its romanised Hindi-English side is labelled `hi_en` and its English side `en`; both are the same conversations in the same order. Tanglish comes from the Tamil part of `community-datasets/offenseval_dravidian` (revision `4739cb65194d4157841510a7c46a6fc87622b764`), keeping the 28,338 of 35,139 comments in which at least 90% of letters are Latin. For each set, turns or comments were joined in dataset order, each followed by a newline, and the token stream was cut into 96 sequences of 512 tokens. A sequence can therefore start or end in the middle of a turn, and its ID is a position in the stream, not a conversation ID. Some Tanglish comments are not Tamil.

**Batching.** Wikipedia and the code-mixed sets ran in batches of 8 with no padding. FLORES ran one sentence at a time. This matters for exact replication: in Exp02 the expert set chosen for about 10% of token-layer pairs depended on which other sequences shared the batch, most likely through BF16 rounding near the top-k boundary. Over whole counts the effect was small, with 0.32% of Wikipedia hits on Qwen3 and 0.46% on Sarvam moving between batched and single runs.

## Gaps

- **Qwen3 Wikipedia is per sequence for English only.** Exp01 saved Qwen3's Wikipedia counts only as fit and evaluation totals per language, and Exp02 reran only the English set, as a bridge. That rerun reproduces Exp01's English totals exactly, and we checked this again when building these files. A shuffled-label control on Qwen3 Wikipedia across languages needs a new GPU run. Sarvam has all ten Wikipedia languages per sequence, and both models have all of FLORES and the code-mixed sets.
- **No co-occurrence counts.** Exp01 recorded which experts were chosen together, on Qwen3 Wikipedia's fit half only. They are not included here.

## Licences

- Wikipedia text is licensed CC BY-SA 3.0 and GFDL. These files contain counts derived from it and no text.
- FLORES-101 is licensed CC BY-SA 4.0. We used it for evaluation only, and no FLORES text is included.
- `festvox/cmu_hinglish_dog` is tagged CC BY-SA 3.0 and GFDL on its dataset card, and `community-datasets/offenseval_dravidian` is tagged CC BY 4.0. No text from either is included.
- Both models are released under Apache 2.0.
