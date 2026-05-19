import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

plt.rcParams.update({'font.size': 14})


# Data for improvements including overall accuracy
attributes = ['Red Hair', 'Yellow Hair', 'Brown Skin', 'Sad Emotion', 'Overall']
# Red hair augmentation improvements
redhair_improvements = [-0.900161 + 0.892110, -0.878378 + 0.891892, 0.003028, 0.008456, 0.00675]
# Yellow hair augmentation improvements
yellowhair_improvements = [-0.900161 + 0.908, -0.878378 + 0.932432, 0.005384, 0.008555, 0.00875]

# Positions
x = np.arange(len(attributes))
width = 0.35  # width of the bars

# Create plot
fig, ax = plt.subplots()

# Add grid and highlight zero line
ax.grid(True, which='major', linewidth=0.75)
ax.axhline(0, color='black', linewidth=1.5)

bars1 = ax.bar(x - width/2, redhair_improvements, width, label='Red Hair Augment', color='#4C72B0')
bars2 = ax.bar(x + width/2, yellowhair_improvements, width, label='Yellow Hair Augment', color='#DD8452')

# Labels and title
ax.set_ylabel('Accuracy Improvement', fontsize=14, fontweight='bold')
# ax.set_title('Red vs. Yellow Hair Augmentation Effects')
ax.set_xticks(x)
ax.set_xticklabels(attributes, fontsize=14, fontweight='bold', rotation=30)
ax.legend(fontsize=14)


# Show value labels on bars
def autolabel_blue(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height*100:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

def autolabel_orange(bars):
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        # Only lower the label for the Yellow Hair bar (index 1)
        if idx == 1:
            xytext = (0, -16)
            va = 'top'
        else:
            xytext = (0, 16)
            va = 'bottom'
        ax.annotate(f'{height*100:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=xytext,
                    textcoords="offset points",
                    ha='center', va=va)

autolabel_blue(bars1)
autolabel_orange(bars2)

# Adjust y-axis limits to ensure labels fit
# ax.set_ylim(-0.01, 0.07)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

plt.tight_layout()
plt.savefig('clip_red_yellow_augment.pdf', format='pdf')