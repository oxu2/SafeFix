import os

files_to_fix = [
    "object_detection/train_yolov8_kitti.py",
    "object_detection/train_yolov8_kitti_generated.py",
    "docs/supp/object_detection/train_yolov8_kitti.py",
    "docs/supp/object_detection/train_yolov8_kitti_generated.py"
]

workspace = "/people/cs/o/oxx220000/safefix"

for fpath in files_to_fix:
    full_path = os.path.join(workspace, fpath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace project=args.project, with project=str(Path(args.project).resolve()),
        content = content.replace("project=args.project,", "project=str(Path(args.project).resolve()),")
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {fpath}")
