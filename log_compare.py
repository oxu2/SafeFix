import re
import os

import argparse

parser = argparse.ArgumentParser(description="Compare hibug vs ours logs")
parser.add_argument('--attributes', type=str, default="simple",
                    help='attributes to detect, e.g. redhair_sademotion, redhair_brownskin, etc.')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip'],
                    help='Model architecture to use (resnet, vit, or clip)')
parser.add_argument('--only_attr', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                    help='Whether to print only metrics for attributes in --attributes.')
args = parser.parse_args()

def parse_scores(log_path):
    """
    Parses a log file and returns two dicts:
    - before: mapping metric name to the accuracy before '->'
    - after: mapping metric name to the accuracy after '->'
    """
    before_scores = {}
    after_scores = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Overall accuracy
            m = re.match(r'Overall accuracy:\s*([0-9.]+)\s*->\s*([0-9.]+)', line)
            if m:
                before_scores['Overall'] = float(m.group(1))
                after_scores['Overall'] = float(m.group(2))
                continue

            # Attribute-specific lines
            m = re.match(r'([^,]+),\s*([^,]+),\s*([0-9.]+)\s*->\s*([0-9.]+)', line)
            if m:
                name = f"{m.group(1).strip()}, {m.group(2).strip()}"
                before_scores[name] = float(m.group(3))
                after_scores[name] = float(m.group(4))
    return before_scores, after_scores

# You can modify the list of added image counts below if needed
for num in [1000, 5000, 10000]:
    ours_log   = f"logs/{args.model}_{args.attributes}_{num}imgs.log"
    hibug_log  = f"logs/hibug_{args.model}_{args.attributes}_{num}imgs.log"

    if not (os.path.exists(ours_log) and os.path.exists(hibug_log)):
        print(f"Skipping {num} imgs: missing log files.")
        continue

    ours_before, ours_after = parse_scores(ours_log)
    hibug_before, hibug_after = parse_scores(hibug_log)

    if args.only_attr:
        include = set()
        for attr in args.attributes.split('_'):
            if attr.endswith('hair'):
                include.add(('hair color', attr[:-4]))
            elif attr.endswith('skin'):
                include.add(('skin', attr[:-4]))
            elif attr.endswith('motion'):
                include.add(('emotion', attr[:-6]))

    print(f"\n=== Comparison for {num} added images ===")
    metrics = sorted(set(hibug_after.keys()) | set(ours_after.keys()))
    for metric in metrics:
        parts = metric.split(', ')
        if len(parts) == 2:
            cat, val = parts
        else:
            cat, val = 'Overall', None
        if args.only_attr and cat != 'Overall':
            # build include set once above
            if (cat, val) not in include:
                continue
        base = ours_before.get(metric, float('nan'))
        h = hibug_after.get(metric, float('nan'))
        o = ours_after.get(metric, float('nan'))
        print(f"{metric:30s}  base = {base:.6f}   hibug = {h:.6f}   ours = {o:.6f}")