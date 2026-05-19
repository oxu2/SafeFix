import os
import re

workspace = "/people/cs/o/oxx220000/safefix"

exclude_dirs = {
    '.git', '.github', '.vscode', '.trellis', '.codex', '__pycache__', '.pytest_cache',
    'logs', 'data', 'save'
}

replacements = [
    # 1. Correct legacy/typo references
    (re.compile(r'save/models/weights/'), 'save/weights/'),
    # 2. Reorganize runs/
    (re.compile(r'(?<!save/)runs/'), 'save/runs/'),
    # 3. Reorganize selection_exports
    (re.compile(r'(?<!save/)\bselection_exports\b'), 'save/selection_exports'),
    # 4. Reorganize weights/
    (re.compile(r'(?<!save/)(?<!models/)weights/'), 'save/weights/')
]

modified_files = 0

for root, dirs, files in os.walk(workspace):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        if file.endswith(('.log', '.pid', '.png', '.jpg', '.pdf', '.xlsx', '.csv', '.pth', '.out', '.pids')):
            continue
        if file == 'rebuttal.md' or file == 'replace_phase2_refs.py':
            continue
            
        file_path = os.path.join(root, file)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            new_content = content
            changed = False
            for pattern, repl in replacements:
                if pattern.search(new_content):
                    new_content = pattern.sub(repl, new_content)
                    changed = True
            
            if changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                rel_path = os.path.relpath(file_path, workspace)
                print(f"Updated: {rel_path}")
                modified_files += 1
        except Exception as e:
            print(f"Error updating {file_path}: {e}")

print(f"Total files updated: {modified_files}")
