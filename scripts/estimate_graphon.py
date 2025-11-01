#!/usr/bin/env python3
import argparse, json, numpy as np
from numpy.linalg import svd

def usvt(adj, eta=0.01):
    n = adj.shape[0]
    # degree-normalize
    deg = adj.sum(1) + 1e-8
    D_inv_sqrt = np.diag(1.0/np.sqrt(deg))
    A = D_inv_sqrt @ adj @ D_inv_sqrt
    U, S, Vt = svd(A, full_matrices=False)
    # estimate sigma from bulk (median of tail)
    tail = S[int(0.8*len(S)):]
    sigma = float(np.median(tail)) if len(tail)>0 else float(np.median(S))
    tau = (2.0 + eta) * np.sqrt(n) * sigma
    keep = S > tau
    S_thr = np.where(keep, S, 0.0)
    return U @ np.diag(S_thr) @ Vt, tau, int(keep.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eta", type=float, default=0.01)
    args = ap.parse_args()

    data = json.load(open(args.graph_json))
    # build adjacency
    N = len(data["neighbors"])
    adj = np.zeros((N, N), dtype=float)
    for i, neigh in enumerate(data["neighbors"]):
        for j, w in neigh:
            adj[i, j] = max(adj[i,j], w)
            adj[j, i] = max(adj[j,i], w)
    W, tau, r = usvt(adj, eta=args.eta)
    json.dump({"tau": tau, "rank": r}, open(args.out,"w"))
    npy_path = args.out.replace(".json",".npy")
    np.save(npy_path, W)
    print(f"Wrote graphon to {args.out} and matrix to {npy_path}")

if __name__ == "__main__":
    main()
