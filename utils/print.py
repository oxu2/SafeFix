import os
import re

LOG_DIR = "logs/imagenet"

# example filename:
# AL_resnet_attr_imagenet_0.1_R0_L0_B0_class_INF_1000.log
fname_pattern = re.compile(
    r"AL_resnet_attr_imagenet_(?P<ratio>[0-9.]+)_"
    r"(?P<config>R\d+_L\d+_B\d+)_class4"
)

# example line:
# Model Test ACC: 0.7507692307692307
acc_pattern = re.compile(r"Model Test ACC:\s*([0-9.]+)")

results = []

for fname in sorted(os.listdir(LOG_DIR)):
    if not fname.endswith(".log"):
        continue

    m = fname_pattern.search(fname)
    if m is None:
        continue

    ratio = m.group("ratio")
    config = m.group("config")

    log_path = os.path.join(LOG_DIR, fname)

    test_acc = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            acc_match = acc_pattern.search(line)
            if acc_match:
                test_acc = float(acc_match.group(1))
                break

    if test_acc is None:
        test_acc = float("nan")

    results.append((float(ratio), config, test_acc))

# print
for ratio, config, acc in results:
    print(f"{ratio}, {config}, {acc}")