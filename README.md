# Graphon-Guided Curriculum for RAG — Reproducible Skeleton

This repository skeleton contains minimal scripts and configs to run:
1. Corpus graph build (ANN k-NN via FAISS)
2. Graphon estimation via USVT (+ optional SAS smoothing placeholder)
3. Difficulty scoring (rarity, boundary entropy proxy, content complexity placeholder)
4. Curriculum scheduling (linear ramp + replay)
5. Attribution evaluation (TRUE-style span-level precision proxy)

See `configs/default.yaml` and `ATTRIBUTION_PROTOCOL.md`.

## Interactive Playground

Want a quick taste of what the curriculum looks like in action? Fire up the Streamlit playground:

```bash
pip install -r requirements.txt
streamlit run apps/graphon_curriculum_playground.py
```

You will be able to:
- Generate synthetic corpus graphs with playful graphon kernels.
- Inspect node-level difficulty scores derived from multiple graph statistics.
- Prototype curriculum schedules with adjustable warmup and replay parameters.

It is a fun way to sanity-check ideas before dropping into the full training pipeline.
