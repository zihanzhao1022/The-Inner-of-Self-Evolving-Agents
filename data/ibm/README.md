# IBM activation-steering demo data

Source: https://github.com/IBM/activation-steering/tree/main/docs/demo-data
Commit: `main` branch (snapshot taken 2026-05-05)
License: Apache-2.0 (see `LICENSE` in this directory; original copyright IBM)

## Files

### `condition_multiple.json`

Paired prompts where each entry contains 6 variants of the same imperative
skeleton, sharing structure but differing in topic/intent:

- `base` — neutral instruction (drawn from Alpaca)
- `legal_opinion` — sensitive-but-legal: legal-advice variant
- `health_consultation` — sensitive-but-legal: medical-advice variant
- `sexual_content` — harmful: adult content
- `hate_speech` — harmful: discriminatory language
- `crime_planning` — harmful: illicit activity planning

Sizes: 700 train + 500 test entries. Each `field` provides 700 prompts when
flattened, totalling 4,200 train prompts across 6 categories.

## Caveats observed in our analysis

1. **Per-class topic spread is huge** — within a single category, prompts
   span all 700 alpaca topics. TF-IDF cosine across same-class prompts is
   ~0.04 (≈ random baseline). This forces the linear probe to learn
   topic-invariant intent directions rather than topic vocabulary.

2. **Within-entry variants share the imperative scaffold** — different
   categories of the same entry have TF-IDF cosine ~0.24 (vs ~0.03 for
   random pairs). This is shared verbs/function-words, not shared topic.

3. **Data quality** — ~2% of entries have at least one near-duplicate pair
   (cosine > 0.7). The worst case (entry 166) is identical between
   `crime_planning` and `health_consultation` except for one capital letter.
   `llm_lens.datasets.load_prompts` filters these by default with a
   configurable cosine threshold.

4. **Category mix is not "1 neutral + 5 harmful"** — `legal_opinion` and
   `health_consultation` are sensitive-but-legal professional domains, not
   harmful. Real category mix is 1 neutral + 2 sensitive-legal + 3 harmful.
