from __future__ import annotations
import argparse
from share.data import load_csv
from share.research import walk_forward_train

def main():
    p = argparse.ArgumentParser(description="Capital-preservation walk-forward training")
    p.add_argument("--csv", required=True)
    p.add_argument("--initial-cash", type=float, default=200_000)
    p.add_argument("--train-days", type=int, default=504)
    p.add_argument("--test-days", type=int, default=126)
    p.add_argument("--purge-days", type=int, default=5)
    args = p.parse_args()
    data = load_csv(args.csv)
    report = walk_forward_train(
        data, initial_cash=args.initial_cash,
        train_days=args.train_days, test_days=args.test_days,
        purge_days=args.purge_days,
    )
    print(report.to_string(index=False))
    report.to_csv("walk_forward_report.csv", index=False)

if __name__ == "__main__":
    main()
