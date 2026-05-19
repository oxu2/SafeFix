import re
import os
import time

ratios = ['0.1', '0.3', '0.5', '0.7', '0.9']
models = {'qwen3': 'Qwen3', 'internvl3': 'InternVL3'}
expected_jobs = len(ratios) * len(models)

print(f"Waiting for {expected_jobs} jobs to complete...")

while True:
    completed = 0
    for model_key in models.keys():
        for ratio in ratios:
            log_file = f"logs/imagenet/aggro_p10_c4_{model_key}_ratio{ratio}.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    if "Best Validation Accuracy" in content or "Test Accuracy:" in content:
                        completed += 1
                        
    if completed == expected_jobs:
        print("All jobs completed. Parsing results...")
        break
        
    print(f"[{time.strftime('%H:%M:%S')}] Completed {completed}/{expected_jobs} jobs. Waiting 60s...")
    time.sleep(60)

results = {}
for model_key, model_name in models.items():
    results[model_name] = {}
    for ratio in ratios:
        log_file = f"logs/imagenet/aggro_p10_c4_{model_key}_ratio{ratio}.log"
        with open(log_file, 'r') as f:
            content = f.read()
            
        val_matches = re.findall(r'Best Validation Accuracy: (\d+\.\d+)', content)
        val_acc = val_matches[-1] if val_matches else "N/A"
        
        test_matches = re.findall(r'Corresponding Test Accuracy: (\d+\.\d+)', content)
        test_acc = test_matches[-1] if test_matches else "N/A"
        
        if test_acc == "N/A":
            alt_test_matches = re.findall(r'Test Accuracy:\s+(\d+\.\d+)', content)
            if alt_test_matches:
                test_acc = alt_test_matches[-1]
                
        results[model_name][ratio] = {"test_acc": test_acc, "val_acc": val_acc}

report_patch = "\n\n### Qwen3 and InternVL3 Results (aggro_exp_p10_c4)\n\n"
report_patch += "This section documents the performance of the generated attribute files from `Qwen3` and `InternVL3` across all standard sub-ratios utilizing the optimal `aggro_exp_p10_c4` setting ($R=0.8, P=10.0, C=4.0$, Entropy=True).\n\n"

for ratio in ratios:
    report_patch += f"#### Sub Ratio = {ratio}\n"
    report_patch += "| Source Model | Test ACC | Val ACC |\n"
    report_patch += "|---|---|---|\n"
    for model_name in ['Qwen3', 'InternVL3']:
        metrics = results[model_name][ratio]
        report_patch += f"| {model_name} | {metrics['test_acc']} | {metrics['val_acc']} |\n"
    report_patch += "\n"

with open("report.md", "a") as f:
    f.write(report_patch)

print("Appended results to report.md successfully.")
