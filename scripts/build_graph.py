#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def embed_corpus(corpus_path: str, model_name: str, normalize: bool = True):
    model = SentenceTransformer(model_name)
    with open(corpus_path) as f:
        docs = [line.strip() for line in f if line.strip()]
    embs = model.encode(docs, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=normalize)
    return docs, embs

def build_hnsw(embs: np.ndarray, M=32, efC=200, efS=64):
    d = embs.shape[1]
    idx = faiss.IndexHNSWFlat(d, M)
    idx.hnsw.efConstruction = efC
    idx.hnsw.efSearch = efS
    idx.add(embs.astype(np.float32))
    return idx

def topk_graph(embs: np.ndarray, idx, top_k=128, min_sim=0.2):
    # cosine sim using inner product on normalized vectors
    I = []
    for i in tqdm(range(embs.shape[0]), desc="top-k"):
        D, J = idx.search(embs[i:i+1].astype(np.float32), top_k+1)
        neighbors = []
        for d, j in zip(D[0], J[0]):
            if j == i: 
                continue
            if d >= min_sim:
                neighbors.append((int(j), float(d)))
        I.append(neighbors)
    return I

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Text file: one document per line")
    ap.add_argument("--out", required=True, help="Output JSON for graph (adj list with weights)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=128)
    ap.add_argument("--min_sim", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--efS", type=int, default=64)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    docs, embs = embed_corpus(args.corpus, args.model, normalize=True)
    idx = build_hnsw(embs, M=args.M, efC=args.efC, efS=args.efS)
    graph = topk_graph(embs, idx, top_k=args.top_k, min_sim=args.min_sim)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"neighbors": graph}, open(args.out,"w"))
    print(f"Wrote graph to {args.out}")

if __name__ == "__main__":
    main()
