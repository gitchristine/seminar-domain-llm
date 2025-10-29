import matplotlib.pyplot as plt
import seaborn as sns

# Data from your training output
epochs = [0, 1, 2, 3, 4]
train_activity_acc = [0.2096, 0.5477, 0.6911, 0.7488, 0.7755]
test_activity_acc = [0.3200, 0.4543, 0.5566, 0.6261, 0.6436]
train_activity_loss = [2.9032, 1.8766, 1.4206, 1.0996, 0.8907]
test_activity_loss = [2.3428, 1.9406, 1.6228, 1.3635, 1.1852]
train_rt_loss = [1.0960, 0.7754, 0.7110, 0.6736, 0.6515]
test_rt_loss = [1.0182, 0.9636, 0.9785, 0.9685, 0.9813]

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RNN Baseline Results on BPI20PrepaidTravelCosts', fontsize=16, fontweight='bold')

# Plot 1: Activity Prediction Accuracy
axes[0, 0].plot(epochs, train_activity_acc, marker='o', linewidth=2, label='Train', color='#2E86AB')
axes[0, 0].plot(epochs, test_activity_acc, marker='s', linewidth=2, label='Test', color='#A23B72')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Next Activity Prediction Accuracy', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

# Plot 2: Activity Prediction Loss
axes[0, 1].plot(epochs, train_activity_loss, marker='o', linewidth=2, label='Train', color='#2E86AB')
axes[0, 1].plot(epochs, test_activity_loss, marker='s', linewidth=2, label='Test', color='#A23B72')
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_title('Next Activity Prediction Loss', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Remaining Time Prediction Loss
axes[1, 0].plot(epochs, train_rt_loss, marker='o', linewidth=2, label='Train', color='#2E86AB')
axes[1, 0].plot(epochs, test_rt_loss, marker='s', linewidth=2, label='Test', color='#A23B72')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Loss (MAE)', fontsize=12)
axes[1, 0].set_title('Remaining Time Prediction Loss', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Summary table
summary_data = [
    ['Metric', 'Final Value'],
    ['Train Activity Acc', f'{train_activity_acc[-1]:.4f}'],
    ['Test Activity Acc', f'{test_activity_acc[-1]:.4f}'],
    ['Train Activity Loss', f'{train_activity_loss[-1]:.4f}'],
    ['Test Activity Loss', f'{test_activity_loss[-1]:.4f}'],
    ['Train RT Loss', f'{train_rt_loss[-1]:.4f}'],
    ['Test RT Loss', f'{test_rt_loss[-1]:.4f}'],
]

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=summary_data[1:], 
                        colLabels=summary_data[0],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)
axes[1, 1].set_title('Final Epoch Metrics', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('rnn_baseline_results.png', dpi=300, bbox_inches='tight')
print("Graph saved as: rnn_baseline_results.png")
plt.show()