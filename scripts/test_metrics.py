import argparse
import json
import os
import sys

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import evaluate_pair


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    pred = cv2.imread(args.pred)
    target = cv2.imread(args.target)

    if pred is None:
        raise RuntimeError(f"Could not read prediction image: {args.pred}")

    if target is None:
        raise RuntimeError(f"Could not read target image: {args.target}")

    if pred.shape[:2] != target.shape[:2]:
        target = cv2.resize(target, (pred.shape[1], pred.shape[0]))

    mask = None

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Could not read mask image: {args.mask}")

    results = evaluate_pair(pred, target, mask)

    print(json.dumps(results, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()