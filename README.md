# Graphon-Guided Curriculum for RAG — Reproducible Skeleton

This repository skeleton contains minimal scripts and configs to run:
1. Corpus graph build (ANN k-NN via FAISS)
2. Graphon estimation via USVT (+ optional SAS smoothing placeholder)
3. Difficulty scoring (rarity, boundary entropy proxy, content complexity placeholder)
4. Curriculum scheduling (linear ramp + replay)
5. Attribution evaluation (TRUE-style span-level precision proxy)

See `configs/default.yaml` and `ATTRIBUTION_PROTOCOL.md`.
