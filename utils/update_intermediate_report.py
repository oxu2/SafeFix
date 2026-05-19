import re
import os

ratios = ['0.1', '0.3', '0.5', '0.7', '0.9']
models = {'qwen3': 'Qwen3', 'internvl3': 'InternVL3'}

results = {}

for ratio in ratios:
    results[ratio] = {}
    for model_key, model_name in models.items():
        log_file = f"logs/imagenet/aggro_p10_c4_{model_key}_ratio{ratio}.log"
        if not os.path.exists(log_file):
            results[ratio][model_name] = {"test_acc": "N/A", "val_acc": "N/A"}
            continue
            
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Try to find the test accuracy line in either the new or older wording
        test_match = re.search(r'(?:Model Test ACC|Held-out test accuracy): (\d+\.\d+)', content)
        test_acc = test_match.group(1) if test_match else "N/A"
        
        # Try to find Epoch [15/15] Val Acc
        val_match = re.search(r'Epoch \[15/15\].*Val Acc: (\d+\.\d+)', content)
        val_acc = val_match.group(1) if val_match else "N/A"
        
        results[ratio][model_name] = {"test_acc": test_acc, "val_acc": val_acc}

# Generate markdown table
table_content = "\n\n### Qwen3 and InternVL3 Results (aggro_exp_p10_c4) - Intermediate\n\n"
for ratio in ratios:
    table_content += f"#### Sub Ratio = {ratio}\n"
    table_content += "| Source Model | Test ACC | Val ACC |\n"
    table_content += "|---|---|---|\n"
    for model_name in ['Qwen3', 'InternVL3']:
        metrics = results[ratio][model_name]
        table_content += f"| {model_name} | {metrics['test_acc']} | {metrics['val_acc']} |\n"
    table_content += "\n"

print(table_content)

# Update report.md
with open("report.md", "r") as f:
    report_content = f.read()

# Check if the intermediate header already exists to replace it
if "### Qwen3 and InternVL3 Results (aggro_exp_p10_c4) - Intermediate" in report_content:
    # Replace existing intermediate block
    import re as regex
    pattern = regex.compile(r'### Qwen3 and InternVL3 Results \(aggro_exp_p10_c4\) - Intermediate.*?(?=\n##|$) ', regex.DOTALL)
    report_content = pattern.sub(table_content.strip() + "\n", report_content)
else:
    # Append to the end of the 2/27/2026 Update section if possible
    if "## Grid Search Results" in report_content:
        parts = report_content.split("## Grid Search Results")
        report_content = parts[0] + table_content + "\n## Grid Search Results" + parts[1]
    else:
        report_content += table_content

with open("report.md", "w") as f:
    f.write(report_content)

print("Report updated.")
