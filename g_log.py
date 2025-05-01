import re
import glob
import argparse
import os

parser = argparse.ArgumentParser(description='Extract final train accuracy from logs.')
parser.add_argument('--model', type=str, default='resnet',
                    help='Model name, e.g., resnet, clip, vit')
parser.add_argument('--num_added_img', type=int, nargs='+', default=[1000, 5000, 10000],
                    help='List of numbers of added images to extract accuracy for')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='Directory containing log files')
parser.add_argument('--attributes', type=str, default="redhair_brownskin",
                    help='attributes to detect, e.g. redhair_sademotion, redhair_brownskin, redhair, brownskin, yellowhair, sademotion, etc.')
args = parser.parse_args()

# Compile the regular expression for the accuracy line
pattern = re.compile(r'Final Epoch Train Accuracy \(gen_only=True\):\s*([0-9]+\.[0-9]+)')

def extract_accuracy_from_log(path):
    """
    Read a log file and extract the final train accuracy.
    """
    acc_value = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # If multiple matches exist, the last one will overwrite
                acc_value = float(match.group(1))
    return acc_value

def main():
    # Iterate over specified numbers of added images and find matching log files
    for num in args.num_added_img:
        pattern = os.path.join(args.log_dir, f'g_{args.model}_{args.attributes}_{num}imgs.log')
        matched_logs = glob.glob(pattern)
        if not matched_logs:
            print(f'No logs found for pattern: {pattern}')
        for log_path in sorted(matched_logs):
            acc = extract_accuracy_from_log(log_path)
            if acc is not None:
                print(f'{log_path}: {acc:.4f}')
            else:
                print(f'{log_path}: accuracy not found')

if __name__ == '__main__':
    main()
