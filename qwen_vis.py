import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def parse_log_file(log_path):
    """Extract results from a log file"""
    with open(log_path, 'r') as f:
        content = f.read()
    
    results = []
    
    # Pattern to find experiments
    exp_pattern = r'=== (.*?) - (.*?) ===.*?Epoch 9:\s+train_next_activity_loss: ([\d.]+).*?train_next_activity_acc: ([\d.]+).*?test_next_activity_loss: ([\d.]+).*?test_next_activity_acc: ([\d.]+).*?train_next_remaining_time_loss: ([\d.]+).*?test_next_remaining_time_loss: ([\d.]+)'
    
    matches = re.finditer(exp_pattern, content, re.DOTALL)
    
    for match in matches:
        method, dataset = match.group(1), match.group(2)
        results.append({
            'Method': method,
            'Dataset': dataset,
            'Train_NA_Loss': float(match.group(3)),
            'Train_NA_Acc': float(match.group(4)),
            'Test_NA_Loss': float(match.group(5)),
            'Test_NA_Acc': float(match.group(6)),
            'Train_RT_Loss': float(match.group(7)),
            'Test_RT_Loss': float(match.group(8))
        })
    
    return results

def create_comparison_plots(df, output_file='results_comparison.png'):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Qwen Model Performance Across Datasets and PEFT Methods', 
                 fontsize=16, fontweight='bold')
    
    datasets = df['Dataset'].unique()
    methods = df['Method'].unique()
    
    colors = sns.color_palette("husl", len(methods))
    method_colors = dict(zip(methods, colors))
    
    # Plot 1: Next Activity Accuracy by Method
    ax = axes[0, 0]
    x = range(len(datasets))
    width = 0.15
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = [method_data[method_data['Dataset'] == d]['Test_NA_Acc'].values[0] 
                 if len(method_data[method_data['Dataset'] == d]) > 0 else 0 
                 for d in datasets]
        ax.bar([xi + i*width for xi in x], values, width, 
               label=method, color=method_colors[method], alpha=0.8)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Next Activity Prediction Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks([xi + width*1.5 for xi in x])
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: Remaining Time Loss by Method
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = [method_data[method_data['Dataset'] == d]['Test_RT_Loss'].values[0] 
                 if len(method_data[method_data['Dataset'] == d]) > 0 else 0 
                 for d in datasets]
        ax.bar([xi + i*width for xi in x], values, width, 
               label=method, color=method_colors[method], alpha=0.8)
    ax.set_ylabel('Test Loss (MAE)', fontsize=11)
    ax.set_title('Remaining Time Prediction Loss', fontsize=12, fontweight='bold')
    ax.set_xticks([xi + width*1.5 for xi in x])
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Train vs Test Accuracy (Overfitting check)
    ax = axes[0, 2]
    for dataset in datasets:
        dataset_data = df[df['Dataset'] == dataset]
        train_acc = dataset_data['Train_NA_Acc'].values
        test_acc = dataset_data['Test_NA_Acc'].values
        ax.scatter(train_acc, test_acc, label=dataset, s=100, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect generalization')
    ax.set_xlabel('Train Accuracy', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Train vs Test Accuracy (Overfitting)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Plot 4: Method comparison heatmap for NA
    ax = axes[1, 0]
    pivot_na = df.pivot(index='Method', columns='Dataset', values='Test_NA_Acc')
    sns.heatmap(pivot_na, annot=True, fmt='.3f', cmap='YlGnBu', 
                ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Next Activity Accuracy Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Plot 5: Method comparison heatmap for RT
    ax = axes[1, 1]
    pivot_rt = df.pivot(index='Method', columns='Dataset', values='Test_RT_Loss')
    sns.heatmap(pivot_rt, annot=True, fmt='.3f', cmap='YlOrRd_r', 
                ax=ax, cbar_kws={'label': 'Loss (lower=better)'})
    ax.set_title('Remaining Time Loss Heatmap', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Plot 6: Summary statistics table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate best method per dataset
    summary_data = []
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        best_na = dataset_df.loc[dataset_df['Test_NA_Acc'].idxmax()]
        best_rt = dataset_df.loc[dataset_df['Test_RT_Loss'].idxmin()]
        summary_data.append([
            dataset,
            f"{best_na['Method']}",
            f"{best_na['Test_NA_Acc']:.3f}",
            f"{best_rt['Method']}",
            f"{best_rt['Test_RT_Loss']:.3f}"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Dataset', 'Best Method\n(NA)', 'Accuracy', 'Best Method\n(RT)', 'Loss'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.25, 0.2, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('Best Performing Methods per Dataset', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.show()

def main():
    # Find the most recent log file
    log_dir = Path('logs')
    log_files = list(log_dir.glob('qwen_3datasets_*.out'))
    
    if not log_files:
        print("No log files found! Make sure experiments have completed.")
        return
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"Processing log file: {latest_log}")
    
    # Parse results
    results = parse_log_file(latest_log)
    
    if not results:
        print("No results found in log file. Make sure experiments completed successfully.")
        return
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('qwen_results_summary.csv', index=False)
    print(f"\nResults saved to: qwen_results_summary.csv")
    
    # Create visualizations
    create_comparison_plots(df)

if __name__ == "__main__":
    main()