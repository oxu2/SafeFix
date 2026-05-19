import glob
import re

log_files = glob.glob("logs/imagenet/resume_lr5_epoch15_*_ratio*.log")
results = {}

strategies_order = [
    "random", "stratified", "coreset", "uncertainty_stratified",
    "diversity", "hierarchical", "hybrid", "attribute"
]

for f in log_files:
    m = re.search(r"resume_lr5_epoch15_(.+)_resnet_imagenet_1000_ratio([\d\.]+)\.log", f)
    if not m:
        continue
    strategy, ratio = m.group(1), m.group(2)

    with open(f, "r") as file:
        content = file.read()

    test_acc_m = re.search(r"(?:Model Test ACC|Held-out test accuracy): ([\d\.]+)", content)
    test_acc = test_acc_m.group(1) if test_acc_m else "N/A"

    val_acc_matches = re.findall(r"Val Acc: ([\d\.]+)", content)
    val_acc = val_acc_matches[-1] if val_acc_matches else "N/A"

    if ratio not in results:
        results[ratio] = {}
    results[ratio][strategy] = {"test": test_acc, "val": val_acc}

for ratio in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
    if ratio not in results:
        continue
    print(f"\n### Sub Ratio = {ratio}")
    print("| Strategy | Test ACC | Val ACC |")
    print("|---|---|---|")

    # Find best test acc for bolding
    best_test = 0.0
    best_strats = []
    for s in strategies_order:
        if s in results[ratio] and results[ratio][s]["test"] != "N/A":
            val = float(results[ratio][s]["test"])
            if val > best_test:
                best_test = val
                best_strats = [s]
            elif val == best_test:
                best_strats.append(s)

    for s in strategies_order:
        if s not in results[ratio]:
            continue
        res = results[ratio][s]
        test_str = res["test"]
        val_str = res["val"]
        if s in best_strats and test_str != "N/A":
            test_str = f"**{test_str}**"
        print(f"| {s} | {test_str} | {val_str} |")
