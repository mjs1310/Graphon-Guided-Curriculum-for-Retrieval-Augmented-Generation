#!/usr/bin/env python3
import argparse, json, numpy as np

def jaccard(a_set, b_set):
    inter = len(a_set & b_set)
    union = len(a_set | b_set) or 1
    return inter/union

def spans(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(0, max(0,len(tokens)-n+1))]

def eval_example(answer, passages, nmin=3, nmax=10, jacc=0.6):
    toks = answer.split()
    cand = []
    for n in range(nmin, min(nmax, len(toks))+1):
        cand += spans(toks, n)
    pos = 0; tot = len(cand)
    passage_spans = []
    for p in passages:
        pt = p.split()
        for n in range(nmin, min(nmax, len(pt))+1):
            passage_spans += spans(pt, n)
    ps_sets = [set(x.split()) for x in passage_spans]
    for s in cand:
        sset = set(s.split())
        if any(jaccard(sset, q) >= jacc for q in ps_sets):
            pos += 1
    prec = pos / (tot or 1)
    return prec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers_json", required=True, help="[{\"answer\": str, \"passages\": [str,...]}, ...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jaccard", type=float, default=0.6)
    args = ap.parse_args()
    data = json.load(open(args.answers_json))
    precs = [eval_example(d["answer"], d["passages"], jacc=args.jaccard) for d in data]
    hallu = [1.0-p for p in precs]
    out = {"attribution_precision_mean": float(np.mean(precs)), "hallucination_rate_mean": float(np.mean(hallu))}
    json.dump(out, open(args.out,"w"))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
