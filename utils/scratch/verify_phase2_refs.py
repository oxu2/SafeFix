import os
import re

workspace = "/people/cs/o/oxx220000/safefix"

exclude_dirs = {
    '.git', '.github', '.vscode', '.trellis', '.codex', '__pycache__', '.pytest_cache',
    'logs', 'data', 'save'
}

patterns = {
    'unreplaced_runs': re.compile(r'(?<!save/)runs/'),
    'unreplaced_selection_exports': re.compile(r'(?<!save/)\bselection_exports\b'),
    'unreplaced_weights': re.compile(r'(?<!save/)(?<!models/)weights/'),
    'legacy_typo_weights': re.compile(r'save/models/weights/')
}

results = []

for root, dirs, files in os.walk(workspace):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        if file.endswith(('.log', '.pid', '.png', '.jpg', '.pdf', '.xlsx', '.csv', '.pth', '.out', '.pids')):
            continue
        if file == 'rebuttal.md' or file == 'verify_phase2_refs.py' or file == 'replace_phase2_refs.py':
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
                            # For unreplaced_weights, if it contains local run directories, skip it
                            if name == 'unreplaced_weights' and ('runs/' in line or 'detect/' in line or 'kitti_yolov8/' in line or 'save_dir' in line):
                                continue
                            results.append((file_path, idx + 1, line, name))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f"Found {len(results)} potential residual references:")
for r in sorted(results):
    rel_path = os.path.relpath(r[0], workspace)
    print(f"{rel_path}:{r[1]}: ({r[3]}) -> {r[2].strip()}")
