# Logging and ML Tracking Guidelines

This project uses a combination of standard console logging and structured artifact saving for machine learning experiments.

---

## Standard Output vs Experiment Tracking

### Console Logging
Use standard `print()` or the `logging` module for real-time console feedback. For training loops, follow the established format to ensure compatibility with parsing tools:

```python
# Required format for utils/plot.py parsing
print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

### Metric Preparation
When implementing new strategies (e.g., Active Learning), prepare metrics for tracking:
- **Training Metrics**: Loss and Accuracy per epoch.
- **AL Metrics**: Selection ratios, difficulty scores, and coverage gaps.
- **Class Distribution**: Log per-class counts for subsets using `_print_class_distribution`.

---

## Diagnostic Reports

Diagnostic reports should be saved to `save/reports/` to maintain a history of model performance and failure analysis.

- **Format**: Use **JSON** for machine-readable data and **Markdown** for human-readable summaries.
- **Content**: Include failure-by-label analysis, prediction correlations, and accuracy comparisons against baselines.
- **Naming**: Use descriptive timestamps and experiment parameters: `save/reports/<dataset>_<model>_<attr>_<n>imgs_<timestamp>/`.

---

## Visualizing Logs

Visualizations are critical for monitoring training progress and diagnosing model behavior.

### Plotting Training Metrics
Save all training plots to `save/plot/`. The standard plotting utility is located in `utils/plot.py`.

- **File Naming**: `save/plot/plot_{model}_{attributes}_{num_added_img}imgs_{timestamp}.png`
- **Content**: Plots should typically include Loss curves and Accuracy (Train vs. Val) curves.

Example usage from `utils/plot.py`:
```bash
python utils/plot.py logs/experiment.log save/plot/training_results.png
```

### Diagnostic Visualizations
Any additional visualizations (e.g., attribute correlation heatmaps) should also be directed to `save/plot/` or included within the diagnostic report directory.

---

## What NOT to Log
- **Large Tensors**: Avoid printing full tensors to the console; log shapes or means instead.
- **Absolute Paths**: Use relative paths from the project root when logging file locations.
- **Secrets**: Never log API keys (e.g., OpenAI keys) or local system credentials.
