#!/usr/bin/env python3
import argparse, json, os, numpy as np

def rarity_from_degree(neighbors):
    deg = np.array([len(n) for n in neighbors], dtype=float)
    r = 1.0 / (1.0 + deg)
    return (r - r.mean())/(r.std()+1e-8)

def boundary_entropy(neighbors, blocks):
    import math
    H = []
    for i, neigh in enumerate(neighbors):
        counts = {}
        for j,_ in neigh:
            b = blocks[j]
            counts[b] = counts.get(b,0)+1
        total = sum(counts.values()) or 1
        ent = -sum((c/total)*math.log(c/total+1e-12) for c in counts.values())
        H.append(ent)
    H = np.array(H)
    return (H - H.mean())/(H.std()+1e-8)

def content_complexity(perplexities):
    P = np.array(perplexities, dtype=float)
    return (P - P.mean())/(P.std()+1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", required=True)
    ap.add_argument("--blocks_json", required=False)
    ap.add_argument("--perplexities_json", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=0.2)
    args = ap.parse_args()

    neighbors = json.load(open(args.graph_json))["neighbors"]
    R = rarity_from_degree(neighbors)

    if args.blocks_json and os.path.exists(args.blocks_json):
        blocks = json.load(open(args.blocks_json))
        B = boundary_entropy(neighbors, blocks)
    else:
        B = np.zeros(len(neighbors))

    if args.perplexities_json and os.path.exists(args.perplexities_json):
        P = content_complexity(json.load(open(args.perplexities_json)))
    else:
        P = np.zeros(len(neighbors))

    D = args.alpha*R + args.beta*B + args.gamma*P
    json.dump({"difficulty": D.tolist()}, open(args.out,"w"))
    print(f"Wrote difficulty scores to {args.out}")

if __name__ == "__main__":
    main()
