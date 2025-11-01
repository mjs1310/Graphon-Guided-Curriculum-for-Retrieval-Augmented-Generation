# REPRODUCIBILITY.md

**Paper:** Graphon-Guided Curriculum for Retrieval-Augmented Generation  
**Commit hash:** <ADD_COMMIT_HASH_HERE>  
**Contact:** <YOUR_EMAIL>

## 1. Code & Environment
- Repository: <URL>
- Commit: <hash>
- Env: Python 3.10; CUDA <ver>; PyTorch <ver>; FAISS <ver>; scikit-learn <ver>.
- Repro container: Dockerfile in `env/`.

## 2. Data & Splits
- RAG data: NQ-Open (version/date). Fixed train/dev/test IDs listed in `artifacts/splits/nq_open/`.
- Proxy: Cora (2708 nodes, 1433 features, 7 classes). Standard or provided random splits (5 seeds).
- TRUE-style evaluation data: <link> (see LICENSES).

## 3. Graph Construction
- Embeddings: <model_name>@<version> (L2-normalized).
- Similarity: cosine; keep top-k neighbors (k=128); min sim 0.2.
- ANN: FAISS HNSW (M=32, efConstruction=200; efSearch=64).

## 4. Graphon Estimation
- USVT: degree-normalized adjacency → SVD; τ = (2+η)·√n·σ (η=0.01); σ from bulk spectrum.
- Retained rank r: # singular values > τ.
- SAS: degree-sort + TV-smoothing (λ=0.1; grid on val).

## 5. Difficulty Score
D_i = α·R_i + β·B_i + γ·C_i with (α,β,γ) = (0.5,0.3,0.2).
- R (rarity): 1/(1+degree_i), z-normalized.
- B (boundary entropy): entropy over block-memberships in 2-hop ego.
- C (content complexity): token perplexity under <LM_name>.

## 6. Curriculum / Scheduler
- Pacing f(t): linear ramp of δ from 10th→90th percentile over epochs 1..T.
- Replay: sample 10% of lowest-D docs each epoch.
- Controls: Reverse/Random curricula.

## 7. Training
- Retriever: <retriever_name>, fixed.
- Generator: <generator_name>, LR 1e-4, batch 64, epochs 6, AdamW, cosine decay, warmup 500, AMP on.

## 8. Evaluation
- Attribution: see ATTRIBUTION_PROTOCOL.md (thresholds, dedup).
- Metrics: EM/F1 + Attribution P/R/F1 + Hallucination Rate.
- CIs: 1000× bootstrap.

## 9. Compute & Carbon
- GPUs: <e.g., A100 80GB> × <n>, hours <h>.
- Energy/Carbon: reported via codecarbon.

## 10. Seeds
- Seeds: 1337, 2025, 31415; report mean ± 95% CI.
