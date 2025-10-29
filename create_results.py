# Manually creating result graphs
import pandas as pd
from pathlib import Path
from datetime import datetime

# Your training results from the terminal
results = [
    {'epoch': 0, 'train_activity_loss': 2.9032, 'train_activity_acc': 0.2096, 
     'test_activity_loss': 2.3428, 'test_activity_acc': 0.3200,
     'train_rt_loss': 1.0960, 'test_rt_loss': 1.0182},
    {'epoch': 1, 'train_activity_loss': 1.8766, 'train_activity_acc': 0.5477,
     'test_activity_loss': 1.9406, 'test_activity_acc': 0.4543,
     'train_rt_loss': 0.7754, 'test_rt_loss': 0.9636},
    {'epoch': 2, 'train_activity_loss': 1.4206, 'train_activity_acc': 0.6911,
     'test_activity_loss': 1.6228, 'test_activity_acc': 0.5566,
     'train_rt_loss': 0.7110, 'test_rt_loss': 0.9785},
    {'epoch': 3, 'train_activity_loss': 1.0996, 'train_activity_acc': 0.7488,
     'test_activity_loss': 1.3635, 'test_activity_acc': 0.6261,
     'train_rt_loss': 0.6736, 'test_rt_loss': 0.9685},
    {'epoch': 4, 'train_activity_loss': 0.8907, 'train_activity_acc': 0.7755,
     'test_activity_loss': 1.1852, 'test_activity_acc': 0.6436,
     'train_rt_loss': 0.6515, 'test_rt_loss': 0.9813},
]

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"BPI20PrepaidTravelCosts_rnn_results_{timestamp}.csv"
filepath = results_dir / filename

df = pd.DataFrame(results)
df.to_csv(filepath, index=False)
print(f"Results saved to: {filepath}")
print(df)