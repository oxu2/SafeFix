import re
import os

def parse_after_scores(log_path):
    """
    Parses a log file and returns a dict mapping each metric name
    to its “after” accuracy (the number after '->').
    """
    scores = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Overall accuracy
            m = re.match(r'Overall accuracy: [0-9.]+ -> ([0-9.]+)', line)
            if m:
                scores['Overall'] = float(m.group(1))
                continue

            # Attribute‐specific lines, e.g. "beard, No, 0.893557 -> 0.891837: declined"
            m = re.match(r'([^,]+),\s*([^,]+),\s*[0-9.]+\s*->\s*([0-9.]+)', line)
            if m:
                name = f"{m.group(1).strip()}, {m.group(2).strip()}"
                scores[name] = float(m.group(3))
    return scores

# list of image‐count suffixes
for num in [1000, 5000, 10000]:
    ours_log   = f"logs/resnet_redhair_brownskin_{num}imgs.log"
    hibug_log  = f"logs/hibug_resnet_redhair_brownskin_{num}imgs.log"

    if not (os.path.exists(ours_log) and os.path.exists(hibug_log)):
        print(f"Skipping {num} imgs: missing log files.")
        continue

    ours_scores  = parse_after_scores(ours_log)
    hibug_scores = parse_after_scores(hibug_log)

    print(f"\n=== Comparison for {num} added images ===")
    for metric in sorted(ours_scores):
        o = ours_scores[metric]
        h = hibug_scores.get(metric, float('nan'))
        print(f"{metric:30s}  ours = {o:.6f}   hibug = {h:.6f}")