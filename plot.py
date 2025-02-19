import sys
import matplotlib.pyplot as plt
import re

def parse_log(log_file):
    """ 解析日志文件，提取 epoch、loss、train acc、val acc """
    epochs, losses, train_accs, val_accs = [], [], [], []
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"Epoch \[(\d+)/\d+\], Loss: ([\d.]+), Train Acc: ([\d.]+), Val Acc: ([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))
                val_accs.append(float(match.group(4)))
    return epochs, losses, train_accs, val_accs

def plot_metrics(epochs, losses, train_accs, val_accs, output_file="training_plot.png"):
    """ 画出 Loss 和 Accuracy 曲线，左右两幅图保存到同一个 PNG 文件中 """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 创建左右两幅子图

    # 左图：Loss 曲线
    axes[0].plot(epochs, losses, "ro-", label="Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True)

    # 右图：Accuracy 曲线
    axes[1].plot(epochs, train_accs, "bo-", label="Train Acc")
    axes[1].plot(epochs, val_accs, "go-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()  # 自动调整子图布局
    plt.savefig(output_file, dpi=300)  # 设置 dpi 提高分辨率
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <log_file> [output_file]")
        sys.exit(1)
    
    log_file = sys.argv[1]  # 从命令行获取日志文件名
    output_file = sys.argv[2] if len(sys.argv) > 2 else "training_plot.png"  # 允许自定义输出文件名
    epochs, losses, train_accs, val_accs = parse_log(log_file)
    plot_metrics(epochs, losses, train_accs, val_accs, output_file)