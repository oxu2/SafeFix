import os
import re

patterns = {
    'log_kitti': re.compile(r'\blog_kitti\b'),
    'log_pose': re.compile(r'\blog_pose\b'),
    'logs_celeba': re.compile(r'\blogs_celeba\b'),
    'logs_imagenet': re.compile(r'\blogs_imagenet\b')
}

workspace = "/people/cs/o/oxx220000/safefix"

exclude_dirs = {
    '.git', '.github', '.vscode', '.trellis', '.codex', '__pycache__', '.pytest_cache',
    'logs', 'data', 'weights', 'outputs', 'runs', 'save', 'save/selection_exports'
}

results = []

for root, dirs, files in os.walk(workspace):
    # modify dirs in place to prune excluded directories
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        if file.endswith(('.log', '.pid', '.png', '.jpg', '.pdf', '.xlsx', '.csv', '.pth', '.out', '.pids')):
            continue
        if file == 'rebuttal.md':
            continue
        file_path = os.path.join(root, file)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for name, pattern in patterns.items():
                if pattern.search(content):
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if pattern.search(line):
                            results.append((file_path, idx + 1, line, name))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f"Found {len(results)} references:")
for r in sorted(results):
    rel_path = os.path.relpath(r[0], workspace)
    print(f"{rel_path}:{r[1]}: ({r[3]}) -> {r[2].strip()}")
