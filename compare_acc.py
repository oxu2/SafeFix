#!/usr/bin/env python3
import re
import sys
import argparse

def parse_log(log_path):
    """
    Parse the log file to extract (attribute, description, accuracy) tuples.
    """
    # 1：ACC: 0.893557  ... attribute "beard", description "No"
    pattern1 = re.compile(
        r'ACC:\s*(?P<acc>\d+\.\d+).*attribute\s+"(?P<attr>[^"]+)",\s*description\s+"(?P<desc>[^"]+)"'
    )
    # 2：The accuracy with "hair color" = "white" is 0.9303...
    pattern2 = re.compile(
        r'The accuracy with\s+"(?P<attr>[^"]+)"\s*=\s*"(?P<desc>[^"]+)"\s*is\s*(?P<acc>\d+\.\d+)'
    )
    pattern_overall = re.compile(r'Model Validation ACC:\s*(?P<acc>\d+\.\d+)')
    overall_acc = None

    results = []
    with open(log_path, 'r') as f:
        for line in f:
            if overall_acc is None:
                m0 = pattern_overall.search(line)
                if m0:
                    overall_acc = float(m0.group('acc'))
            m1 = pattern1.search(line)
            if m1:
                groups = m1.groupdict()
                results.append({
                    'attribute': groups['attr'],
                    'description': groups['desc'],
                    'accuracy': float(groups['acc'])
                })
                continue

            m2 = pattern2.search(line)
            if m2:
                groups = m2.groupdict()
                results.append({
                    'attribute': groups['attr'],
                    'description': groups['desc'],
                    'accuracy': float(groups['acc'])
                })

    return results, overall_acc

def main():
    """
    Compare accuracy metrics between two log files.
    Usage: analysis.py [--orig ORIG_LOG] [--new NEW_LOG]
    """
    parser = argparse.ArgumentParser(description="Compare accuracy metrics between two log files.")
    parser.add_argument('--orig', '-o', default='logs/stat_origin_resnet18_old.log',
                        help='Path to original log file')
    parser.add_argument('--new', '-n', default='logs/stat_redhair_resnet18_0417122501.log',
                        help='Path to new log file')
    args = parser.parse_args()

    orig_file = args.orig
    new_file = args.new

    # extract entries from both files
    orig_entries, orig_overall = parse_log(orig_file)
    new_entries, new_overall = parse_log(new_file)

    # build lookup maps for quick access
    orig_map = {(e['attribute'], e['description']): e['accuracy'] for e in orig_entries}
    new_map = {(e['attribute'], e['description']): e['accuracy'] for e in new_entries}

    # compare overall model validation accuracy
    if orig_overall is not None and new_overall is not None:
        if new_overall > orig_overall:
            overall_status = 'improved'
        elif new_overall < orig_overall:
            overall_status = 'declined'
        else:
            overall_status = 'same'
        print(f'Overall accuracy: {orig_overall:.6f} -> {new_overall:.6f}: {overall_status}')

    # compare and print results
    for (attribute, description), orig_acc in orig_map.items():
        new_acc = new_map.get((attribute, description))
        if new_acc is None:
            # print missing entries in new log
            print(f'{attribute}, {description}: not found in new log')
            continue
        # determine status
        if new_acc > orig_acc:
            status = 'improved'
        elif new_acc < orig_acc:
            status = 'declined'
        else:
            status = 'same'
        print(f'{attribute}, {description}, {orig_acc:.6f} -> {new_acc:.6f}: {status}')

if __name__ == '__main__':
    main()