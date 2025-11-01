# ATTRIBUTION_PROTOCOL.md

## Matching
- Tokenize generated answers and retrieved passages.
- Candidate spans: answer n-grams (n=3..10).
- A span is **attributed** if any retrieved passage contains a span with:
  - Jaccard ≥ 0.6 **or**
  - cosine(sim(embedding(answer_span), embedding(passage_span))) ≥ 0.85 within ±50 tokens.

## Metrics
- Attribution Precision / Recall / F1
- Hallucination Rate = 1 − Precision

## Aggregation (TRUE-style)
- An answer is faithful if each distinct factual claim has ≥1 attributed support span.
- Thresholds and windows are published in this file and config YAMLs.
