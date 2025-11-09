import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns

# List of CSV files to load
CSV_FILES = glob.glob('results/*_replication_results.csv')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_combine_results(csv_files):
    """
    Load all CSV files and combine them into a single dataframe
    """
    dfs = []
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded: {file} ({len(df)} rows)")
        else:
            print(f"Warning: File not found - {file}")
    
    if not dfs:
        raise ValueError("No valid CSV files found!")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    return combined_df

def parse_backbone_info(df):
    """
    Extract model name and PEFT method from backbone string
    """
    df = df.copy()
    
    # Extract PEFT method from brackets
    df['PEFT_Method'] = df['Backbone'].str.extract(r'\[(.*?)\]')[0]
    
    # Extract base model name (before brackets)
    df['Model'] = df['Backbone'].str.extract(r'^(.*?)\s*\[')[0]
    
    # Classify PEFT type
    df['PEFT_Type'] = df['PEFT_Method'].apply(lambda x: 
        'LoRA' if pd.notna(x) and 'LoRA' in x else 'Freezing')
    
    return df

def get_best_configs_per_model_dataset(df):
    """
    For each dataset and model, get the configuration with:
    - Best NA Accuracy
    - Best RT MSE (lowest)
    - Report runtime of best configuration
    """
    df = parse_backbone_info(df)
    
    best_results = []
    
    for dataset in sorted(df['Dataset'].unique()):
        for model in sorted(df['Model'].unique()):
            subset = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            
            if len(subset) == 0:
                continue
            
            # Find best configuration for each metric
            best_na_idx = subset['NA Acc.'].idxmax()
            best_rt_idx = subset['RT MSE'].idxmin()
            
            best_na_row = subset.loc[best_na_idx]
            best_rt_row = subset.loc[best_rt_idx]
            
            # If same config is best for both, use that
            # Otherwise, report the config that's best for NA (primary metric)
            result = {
                'Dataset': dataset,
                'Backbone': model,
                'NA Acc.': best_na_row['NA Acc.'],
                'RT MSE': best_rt_row['RT MSE'],
                '# params (%trainable)': f"{model} (varies)",  # Update if you have this info
                'Runtime (hours)': round(best_na_row['Runtime (h)'], 3),
                'Best NA Config': best_na_row['PEFT_Method'],
                'Best RT Config': best_rt_row['PEFT_Method']
            }
            best_results.append(result)
    
    return pd.DataFrame(best_results)

# ============================================================================
# TABLE 2 GENERATION
# ============================================================================

def create_table_2(df, save_path='table2.csv'):
    """
    Create Table 2 showing best results per model per dataset
    """
    table_df = get_best_configs_per_model_dataset(df)
    
    # Print formatted table
    print("\n" + "="*100)
    print("TABLE 2: Top NA accuracy and RT MSE per LLM with PEFT")
    print("="*100)
    
    # Format for better display
    display_df = table_df[['Dataset', 'Backbone', 'NA Acc.', 'RT MSE', 
                           'Runtime (hours)']].copy()
    display_df['NA Acc.'] = display_df['NA Acc.'].round(4)
    display_df['RT MSE'] = display_df['RT MSE'].round(4)
    
    print(display_df.to_string(index=False))
    print("="*100)
    
    # Save to CSV
    table_df.to_csv(save_path, index=False)
    print(f"\nTable saved to: {save_path}")
    
    # Also save a LaTeX version
    latex_path = save_path.replace('.csv', '.tex')
    with open(latex_path, 'w') as f:
        f.write(display_df.to_latex(index=False, float_format="%.4f"))
    print(f"LaTeX table saved to: {latex_path}")
    
    return table_df

# ============================================================================
# VIOLIN PLOTS
# ============================================================================

def create_violin_plots_figure2(df, save_path='violin_plots.png'):
    """
    Create violin plots similar to Figure 2 in the paper
    (a) LoRA vs Freezing comparison
    (b) Detailed freezing configurations
    """
    df = parse_backbone_info(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color palettes
    peft_colors = ['#7570b3', '#66a61e']  # Purple for LoRA, Green for Freezing
    
    # ============= FIGURE 2a: LoRA vs Freezing =============
    
    # NA Accuracy
    ax = axes[0, 0]
    sns.violinplot(data=df, x='Model', y='NA Acc.', hue='PEFT_Type',
                   ax=ax, palette=peft_colors, inner='box', cut=0)
    ax.set_ylabel('NA Acc.', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('(a) NA Accuracy by PEFT Method', fontsize=13, fontweight='bold')
    ax.legend(title='PEFT', fontsize=10, title_fontsize=11)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # RT MSE
    ax = axes[1, 0]
    sns.violinplot(data=df, x='Model', y='RT MSE', hue='PEFT_Type',
                   ax=ax, palette=peft_colors, inner='box', cut=0)
    ax.set_ylabel('RT MSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('', fontsize=12)
    ax.legend(title='PEFT', fontsize=10, title_fontsize=11)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # ============= FIGURE 2b: Detailed Freezing Configs =============
    
    freezing_df = df[df['PEFT_Type'] == 'Freezing'].copy()
    
    if len(freezing_df) > 0:
        # NA Accuracy - detailed
        ax = axes[0, 1]
        sns.violinplot(data=freezing_df, x='Model', y='NA Acc.',
                       hue='PEFT_Method', ax=ax, inner='box', cut=0)
        ax.set_ylabel('NA Acc.', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_title('(b) NA Accuracy by Freezing Configuration', 
                    fontsize=13, fontweight='bold')
        ax.legend(title='Freezing Config', fontsize=8, title_fontsize=9,
                 loc='lower right', ncol=1)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
        
        # RT MSE - detailed
        ax = axes[1, 1]
        sns.violinplot(data=freezing_df, x='Model', y='RT MSE',
                       hue='PEFT_Method', ax=ax, inner='box', cut=0)
        ax.set_ylabel('RT MSE', fontsize=12, fontweight='bold')
        ax.set_xlabel('', fontsize=12)
        ax.legend(title='Freezing Config', fontsize=8, title_fontsize=9,
                 loc='upper right', ncol=1)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
    else:
        for ax in [axes[0, 1], axes[1, 1]]:
            ax.text(0.5, 0.5, 'No freezing configurations found',
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Fig. 2: PEFT Method Comparison Across Models', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nViolin plots saved to: {save_path}")
    
    return fig



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("CREATING TABLE 2 AND LOSS CURVES")
    print("="*100)
    
    # Load all results
    print("\nLoading results...")
    df = load_and_combine_results(CSV_FILES)
    
    # Show summary
    print(f"\nDatasets: {sorted(df['Dataset'].unique())}")
    print(f"Unique backbones: {len(df['Backbone'].unique())}")
    
    # Create Table 2
    print("\n" + "-"*100)
    print("GENERATING TABLE 2...")
    print("-"*100)
    # table_df = create_table_2(df, save_path='results/table2_results.csv')
    
    # Create Violin Plots (Figure 2)
    # Create violin plots
    print("\nGenerating violin plots...")
    create_violin_plots_figure2(df, save_path='results/violin_plots_figure2.png')
    
    print("\n" + "="*100)
    print("COMPLETE!")
    print("="*100)
    print("\nGenerated files:")
    print("  - table2_results.csv")
    print("  - violin_plots_figure2.png")