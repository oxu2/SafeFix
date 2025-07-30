#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

def is_image(fname):
    return fname.lower().endswith(('.jpg','.jpeg','.png','.bmp'))

def count_dir(dir_path):
    return sum(1 for f in os.listdir(dir_path) if is_image(f))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='imagenet7',
                   help='Root folder containing class subfolders (or train/val).')
    args = p.parse_args()

    root = Path(args.data_dir)
    train_root = root / 'train'
    val_root   = root / 'val'

    use_split = train_root.is_dir() and val_root.is_dir()
    print(f"{'Using existing train/val split.' if use_split else 'No train/val dirs found; showing 80:20 split suggestion.'}\n")

    classes = sorted([d.name for d in root.iterdir() if d.is_dir() and d.name not in ('train','val')])
    overall = {'train': 0, 'val': 0, 'total': 0}

    for cls in classes:
        if use_split:
            n_train = count_dir(train_root/cls)
            n_val   = count_dir(val_root/cls)
            total   = n_train + n_val
        else:
            total = count_dir(root/cls)
            n_train = int(total * 0.8)
            n_val   = total - n_train

        overall['train'] += n_train
        overall['val']   += n_val
        overall['total'] += total

        print(f"{cls:15s}  total: {total:4d}   train: {n_train:4d}   val: {n_val:4d}")

    print("\nOverall:", 
          f"total={overall['total']},",
          f"train={overall['train']} ({overall['train']/overall['total']:.1%}),",
          f"val={overall['val']} ({overall['val']/overall['total']:.1%})")

if __name__ == '__main__':
    main()