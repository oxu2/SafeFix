import re
import os
import glob

import argparse

parser = argparse.ArgumentParser(description="Compare hibug vs ours logs")
parser.add_argument('--attributes', type=str, default="simple",
                    help='attributes to detect, e.g. redhair_sademotion, redhair_brownskin, etc.')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip'],
                    help='Model architecture to use (resnet, vit, or clip)')
parser.add_argument('--only_attr', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                    help='Whether to print only metrics for attributes in --attributes.')
parser.add_argument('--prefix', type=str, default='g',
                    help='Prefix for the "ours" log files (e.g. "g" or "g_simple")')
parser.add_argument('--base_prefix', type=str, default='g',
                    help='Prefix for the base log files (default: g)')
args = parser.parse_args()
# Allow multiple attribute sets, comma-separated
attr_sets = args.attributes.split(',')

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
for attributes in attr_sets:
    for num in [1000, 5000, 10000]:
        # Resolve ours_log via glob (allow variations in naming)
        ours_pattern = f"logs/{args.prefix}_{args.model}_{attributes}_{num}imgs*.log"
        ours_matches = glob.glob(ours_pattern)
        if ours_matches:
            ours_log = ours_matches[0]
        else:
            ours_log = f"logs/{args.prefix}_{args.model}_{attributes}_{num}imgs_best.log"

        # Resolve base_log via glob (allow variations in naming)
        base_pattern = f"logs/{args.base_prefix}_{args.model}_{attributes}_{num}imgs*.log"
        base_matches = glob.glob(base_pattern)
        if base_matches:
            base_log = base_matches[0]
        else:
            base_log = f"logs/{args.base_prefix}_{args.model}_{attributes}_{num}imgs_best.log"

        if not (os.path.exists(ours_log) and os.path.exists(base_log)):
            print(f"Skipping {attributes} / {num} imgs: missing log files for prefixes '{args.prefix}' or '{args.base_prefix}'.")
            continue

        ours_before, ours_after = parse_scores(ours_log)
        base_before, base_after = parse_scores(base_log)

        # If comparing gen_only logs (gen_only prefix "g"), only print final train accuracy
        if args.base_prefix == 'g':
            # Pattern to extract final training accuracy
            train_acc_pattern = re.compile(r'Final Epoch Train Accuracy.*:\s*([0-9.]+)')
            def extract_train_acc(path):
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        m = train_acc_pattern.search(line)
                        if m:
                            return float(m.group(1))
                return float('nan')
            base_acc = extract_train_acc(base_log)
            ours_acc = extract_train_acc(ours_log)
            # Rename labels for clarity
            base_label = "g_ours"
            prefix_label = "g_hibug_simple" if args.prefix == "g_simple" else args.prefix
            print(f"\n=== Final Epoch Train Accuracy ({prefix_label} vs {base_label}) for '{attributes}' with {num} added images ===")
            print(f"{base_label} = {base_acc:.4f}, {prefix_label} = {ours_acc:.4f}")
            continue

        if args.only_attr:
            include = set()
            for attr in attributes.split('_'):
                if attr.endswith('hair'):
                    include.add(('hair color', attr[:-4]))
                elif attr.endswith('skin'):
                    include.add(('skin', attr[:-4]))
                elif attr.endswith('motion'):
                    include.add(('emotion', attr[:-6]))

        print(f"\n=== Comparison ({args.prefix} vs {args.base_prefix}) for '{attributes}' with {num} added images ===")
        metrics = sorted(set(base_after.keys()) | set(ours_after.keys()))
        for metric in metrics:
            parts = metric.split(', ')
            if len(parts) == 2:
                cat, val = parts
            else:
                cat, val = 'Overall', None
            if args.only_attr and cat != 'Overall':
                if (cat, val) not in include:
                    continue
            b = base_after.get(metric, float('nan'))
            o = ours_after.get(metric, float('nan'))
            print(f"{metric:30s}  {args.base_prefix} = {b:.6f}   {args.prefix} = {o:.6f}")