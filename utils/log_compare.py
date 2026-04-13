import re
import os
import pandas as pd
import numpy as np

# Define attribute groups for reporting
ATTRIBUTE_GROUPS = [
    'redhair_brownskin_sademotion',
    'redhair',
    'brownskin',
    'sademotion',
    'redhair_brownskin',
    'redhair_sademotion'
]


import argparse

parser = argparse.ArgumentParser(description="Compare safefix vs ours logs")
parser.add_argument('--attributes', type=str, default=None,
                    help='attributes to detect, e.g. redhair_sademotion, redhair_brownskin, etc.')
parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vit', 'clip'],
                    help='Model architecture to use (resnet, vit, or clip)')
parser.add_argument('--only_attr', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                    help='Whether to print only metrics for attributes in --attributes.')
parser.add_argument('--excel', type=lambda x: x.lower() in ['true','1','yes'], default=True,
                    help='Whether to save output to Excel (True/False).')
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet','celeba'],
                    help='Dataset to use (imagenet or celeba).')
args = parser.parse_args()

# Select log directory based on dataset
if args.dataset == 'imagenet':
    log_dir = 'logs_imagenet'
else:
    log_dir = 'logs'

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

# choose image counts based on dataset
if args.dataset == 'imagenet':
    nums = [100, 500, 1000]
else:
    nums = [1000, 5000, 10000]
results = []
for num in nums:
    # determine attribute part for ours_log
    if args.dataset == 'imagenet':
        # groups already set above in excel branch, but ensure here too
        if args.model in ['resnet', 'vit']:
            attr_part = 'pinkcolor_fabrictexture'
        elif args.model == 'clip':
            attr_part = 'orangecolor_fabrictexture'
        else:
            attr_part = args.attributes or ''
    else:
        attr_part = args.attributes or ''
    ours_log = f"{log_dir}/{args.model}_{attr_part}_{num}imgs.log"
    
    if args.dataset == 'imagenet':
        safefix_log = f"{log_dir}/simple_{args.model}_{attr_part}_{num}imgs.log"
        # example: logs_imagenet/safefix_resnet_simple_pinkcolor_fabrictexture_1000imgs.log
        print(f"safefix log: {safefix_log}")
    else:
        safefix_log = f"{log_dir}/safefix_{args.model}_simple_{num}imgs_last.log"
    class_log = f"{log_dir}/safefix_{args.model}_class_{num}imgs.log"

    # require ours_log always
    if not os.path.exists(ours_log):
        print(f"Skipping {num} imgs: missing ours log.")
        continue
    # parse ours
    ours_before, ours_after = parse_scores(ours_log)
    # parse safefix and class logs, but continue even if missing
    try:
        safefix_before, safefix_after = parse_scores(safefix_log)
    except FileNotFoundError:
        safefix_before, safefix_after = {}, {}
    try:
        class_before, class_after = parse_scores(class_log)
    except FileNotFoundError:
        class_before, class_after = {}, {}

    # if args.only_attr:
    #     include = set()
    #     for attr in args.attributes.split('_'):
    #         if attr.endswith('hair'):
    #             include.add(('hair color', attr[:-4]))
    #         elif attr.endswith('skin'):
    #             include.add(('skin', attr[:-4]))
    #         elif attr.endswith('motion'):
    #             include.add(('emotion', attr[:-6]))

    base_val = ours_before.get('Overall', float('nan'))
    safefix_val = safefix_after.get('Overall', float('nan'))
    base_val *= 100
    safefix_val *= 100
    if args.excel:
        # select attribute groups based on dataset and model
        if args.dataset == 'imagenet':
            if args.model in ['resnet', 'vit']:
                groups = ['pinkcolor_fabrictexture']
            elif args.model == 'clip':
                groups = ['orangecolor_fabrictexture']
            else:
                groups = ATTRIBUTE_GROUPS
        else:
            groups = ATTRIBUTE_GROUPS
        class_val = class_after.get('Overall', float('nan')) * 100
        # extract Overall ACC for each attribute group from its own log file
        attr_accs = {}
        for attr in groups:
            log_path = f"{log_dir}/{args.model}_{attr}_{num}imgs.log"
            if os.path.exists(log_path):
                _, scores = parse_scores(log_path)
                attr_accs[attr] = scores.get('Overall', float('nan')) * 100
            else:
                attr_accs[attr] = float('nan')
        # avg = np.mean(list(attr_accs.values()))
        results.append({
            'num_images': num,
            'base': base_val,
            'class': class_val,
            'safefix': safefix_val,
            **attr_accs,
            # 'average': avg
        })
    else:
        # original print block (indent under this else)
        print(f"\n=== Comparison for {num} added images ===")
        metrics = sorted(set(safefix_after.keys()) | set(ours_after.keys()))
        for metric in metrics:
            parts = metric.split(', ')
            if len(parts) == 2:
                cat, val = parts
            else:
                cat, val = 'Overall', None
            if args.only_attr and cat != 'Overall':
                # build include set once above
                # if (cat, val) not in include:
                #     continue
                # Instead, skip metrics not in groups if only_attr is True
                found = False
                for group in groups:
                    if val and val in group:
                        found = True
                        break
                if not found:
                    continue
            base = ours_before.get(metric, float('nan'))
            h = safefix_after.get(metric, float('nan'))
            o = ours_after.get(metric, float('nan'))
            print(f"{metric:30s}  base = {base:.6f}   safefix = {h:.6f}   ours = {o:.6f}")

# After the loop, save Excel if requested
if args.excel:
    df = pd.DataFrame(results).set_index('num_images')
    out_name = f"{args.dataset}_{args.model}_comparison.xlsx"
    try:
        df.to_excel(out_name)
        print(f"Saved Excel report to {out_name}")
    except ModuleNotFoundError:
        csv_name = out_name.replace('.xlsx', '.csv')
        df.to_csv(csv_name)
        print(f"openpyxl not installed, saved CSV report to {csv_name}")