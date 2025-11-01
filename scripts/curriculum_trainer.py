#!/usr/bin/env python3
import argparse, json, random, numpy as np

def schedule_indices(D, epoch, total_epochs, replay_pct=0.1):
    lo = np.percentile(D, 10)
    hi = np.percentile(D, 90)
    delta = lo + (hi-lo) * (epoch/(total_epochs-1))
    keep = np.where(D <= delta)[0].tolist()
    replay = np.where(D <= lo)[0].tolist()
    k = max(1, int(len(keep)*replay_pct))
    if replay:
        keep += random.sample(replay, min(k, len(replay)))
    return sorted(set(keep))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty_json", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--replay_pct", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    D = np.array(json.load(open(args.difficulty_json))["difficulty"])
    schedule = {}
    for e in range(args.epochs):
        idx = schedule_indices(D, e, args.epochs, args.replay_pct)
        schedule[e] = idx
    json.dump(schedule, open(args.out,"w"))
    print(f"Wrote curriculum index schedule to {args.out}")

if __name__ == "__main__":
    main()
