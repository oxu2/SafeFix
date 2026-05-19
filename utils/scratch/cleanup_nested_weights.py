import os
import re

workspace = "/people/cs/o/oxx220000/safefix"

# 1. .gitignore
gitignore_path = os.path.join(workspace, '.gitignore')
if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r') as f:
        lines = f.read().splitlines()
    new_lines = []
    for line in lines:
        if 'save/runs/' in line and 'save/weights/' in line:
            line = line.replace('save/weights/', 'weights/')
        new_lines.append(line)
    with open(gitignore_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    print("Cleaned .gitignore")

# 2. utils/stat_kitti_detection.py
stat_path = os.path.join(workspace, 'utils/stat_kitti_detection.py')
if os.path.exists(stat_path):
    with open(stat_path, 'r') as f:
        content = f.read()
    content = content.replace('save/runs/kitti_yolov8/.../save/weights/best.pt', 'save/runs/kitti_yolov8/.../weights/best.pt')
    with open(stat_path, 'w') as f:
        f.write(content)
    print("Cleaned utils/stat_kitti_detection.py")

# 3. train_yolov8_kitti_generated.py
for filepath in ['object_detection/train_yolov8_kitti_generated.py', 'docs/supp/object_detection/train_yolov8_kitti_generated.py']:
    full_path = os.path.join(workspace, filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()
        content = content.replace('"save/weights/best.pt"', '"weights/best.pt"')
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Cleaned {filepath}")

# 4. docs/juno_quickstart.md
quickstart_path = os.path.join(workspace, 'docs/juno_quickstart.md')
if os.path.exists(quickstart_path):
    with open(quickstart_path, 'r') as f:
        content = f.read()
    content = content.replace('save/runs/<run-name>/save/weights/best.pt', 'save/runs/<run-name>/weights/best.pt')
    with open(quickstart_path, 'w') as f:
        f.write(content)
    print("Cleaned docs/juno_quickstart.md")

# 5. .handoff.md and report.md
for filepath in ['.handoff.md', 'report.md']:
    full_path = os.path.join(workspace, filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()
        # Find any occurrence of save/runs/.../save/weights and replace it with save/runs/.../weights
        content = re.sub(r'(save/runs/[^/\s\)]+/save/weights/)', lambda m: m.group(1).replace('save/weights/', 'weights/'), content)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Cleaned {filepath}")
