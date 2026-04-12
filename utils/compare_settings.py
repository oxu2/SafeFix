"""
Setting comparison between bash script and Python demo
"""

def compare_settings():
    """Compare settings between different execution methods"""
    
    print("=== Settings Comparison ===\n")
    
    settings_comparison = {
        'Parameter': ['batch_size', 'learning_rate', 'num_epochs', 'initial_labeled_size', 'al_rounds', 'samples_per_round'],
        'Bash Script (run_active_learning.sh)': [32, '1e-4', 10, 1000, 5, 500],
        'Python Demo (tests/demo_active_learning.py)': [16, '1e-4', 5, 500, 3, 200],
        'Enhanced AL (enhanced_active_learning.py)': [16, '1e-4', 3, 300, 3, 100]
    }
    
    print(f"{'Parameter':<20} {'Bash Script':<15} {'Python Demo':<15} {'Enhanced AL':<15}")
    print("-" * 70)
    
    for i, param in enumerate(settings_comparison['Parameter']):
        bash_val = settings_comparison['Bash Script (run_active_learning.sh)'][i]
        demo_val = settings_comparison['Python Demo (tests/demo_active_learning.py)'][i] 
        enhanced_val = settings_comparison['Enhanced AL (enhanced_active_learning.py)'][i]
        
        print(f"{param:<20} {bash_val:<15} {demo_val:<15} {enhanced_val:<15}")
    
    print("\n=== Data Split (Same for all) ===")
    print("All scripts use the same load_celeba_data() function:")
    print("- Training set: 80,000 samples (from train_idx.npy)")
    print("- Validation set: 10,000 samples (random within last 100,000; cached per seed)")
    print("- Test set: 10,000 samples (disjoint from validation, same pool)")
    print("- Unlabeled set: 80,000 samples (remaining from the 100,000-sample pool)")
    print("- Total images: ~200,000 (from CelebA dataset)")
    
    print("\n=== Key Differences ===")
    print("1. Bash script: Full-scale training (more epochs, larger batches, more AL rounds)")
    print("2. Python demo: Quick demonstration (fewer epochs, smaller batches, fewer AL rounds)")
    print("3. Enhanced AL: Strategy comparison (minimal settings for testing)")
    
    print("\n=== Recommendation ===")
    print("- Use Python demo for quick testing and verification")
    print("- Use bash script for production/research training")
    print("- Use enhanced AL for comparing different strategies")

if __name__ == "__main__":
    compare_settings()
