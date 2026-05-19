#!/usr/bin/env python3
"""
Compare Active Learning strategy results
比较Active Learning策略结果
"""

import re
import glob
import pandas as pd

def parse_log_file(log_path):
    """Parse log file to extract final accuracy and training time"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract final validation accuracy
        val_acc_matches = re.findall(r'Val Acc: (\d+\.\d+)', content)
        final_val_acc = float(val_acc_matches[-1]) if val_acc_matches else None
        
        # Extract training time
        time_match = re.search(r'Total training time for \d+ epochs: (\d+\.\d+) seconds', content)
        training_time = float(time_match.group(1)) if time_match else None
        
        # Extract strategy name from filename
        strategy = log_path.split('AL_')[1].split('_')[0]
        
        return {
            'strategy': strategy,
            'final_val_acc': final_val_acc,
            'training_time': training_time,
            'log_file': log_path
        }
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None

def main():
    import sys
    
    # Parse command line arguments for ratio
    ratio = "0.5"  # default
    if len(sys.argv) > 1:
        ratio = sys.argv[1]
    
    ratio_str = f"ratio{ratio}"
    
    # Find all AL log files with the specified ratio
    log_files = glob.glob(f'logs/imagenet/AL_*_resnet_imagenet_1000_{ratio_str}.log')
    
    if not log_files:
        print(f"No log files found for ratio {ratio}. Pattern: AL_*_resnet_imagenet_1000_{ratio_str}.log")
        return
    
    results = []
    for log_file in log_files:
        result = parse_log_file(log_file)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found. Make sure tests have completed.")
        return
    
    # Create DataFrame and sort by accuracy
    df = pd.DataFrame(results)
    df = df.sort_values('final_val_acc', ascending=False)
    
    print(f"\n=== Active Learning Strategy Comparison (sub_ratio={ratio}) ===")
    print(f"{'Strategy':<12} {'Val Acc':<10} {'Time (s)':<10} {'Improvement':<12}")
    print("-" * 50)
    
    baseline_acc = df[df['strategy'] == 'random']['final_val_acc'].values
    baseline_acc = baseline_acc[0] if len(baseline_acc) > 0 else None
    
    for _, row in df.iterrows():
        improvement = ""
        if baseline_acc and row['final_val_acc']:
            imp = ((row['final_val_acc'] - baseline_acc) / baseline_acc) * 100
            improvement = f"{imp:+.2f}%"
        
        print(f"{row['strategy']:<12} {row['final_val_acc']:<10.4f} {row['training_time']:<10.1f} {improvement:<12}")
    
    print(f"\nBaseline (random): {baseline_acc:.4f}")
    print("Best strategy:", df.iloc[0]['strategy'], f"({df.iloc[0]['final_val_acc']:.4f})")

if __name__ == "__main__":
    main()
