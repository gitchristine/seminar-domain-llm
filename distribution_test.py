import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from skpm.event_logs import (
    BPI12, BPI17, BPI20PrepaidTravelCosts,
    BPI20TravelPermitData, BPI20RequestForPayment
)
from skpm.feature_extraction import TimestampExtractor

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_and_preprocess(log_class, log_name):
    """Load and preprocess a single event log"""
    log = log_class()
    df = log.dataframe
    
    # Basic preprocessing
    df = df.loc[:, ["case:concept:name", "concept:name", "time:timestamp"]]
    cases_to_drop = df.groupby("case:concept:name").size() > 2
    cases_to_drop = cases_to_drop[cases_to_drop].index
    df = df[df["case:concept:name"].isin(cases_to_drop)]
    df = df.sort_values(by=["case:concept:name", "time:timestamp"])
    
    # Extract time features
    ts = TimestampExtractor(
        case_features=["accumulated_time", "remaining_time"],
        event_features="all",
        time_unit="d"
    )
    df[ts.get_feature_names_out()] = ts.fit_transform(df)
    
    df = df.rename(columns={
        "case:concept:name": "case_id",
        "concept:name": "activity"
    })
    
    df['dataset'] = log_name
    return df

def compute_dataset_statistics(datasets_dict):
    """Compute key statistics for each dataset"""
    stats_list = []
    
    for name, df in datasets_dict.items():
        # Basic statistics
        n_cases = df['case_id'].nunique()
        n_events = len(df)
        n_activities = df['activity'].nunique()
        
        # Trace length statistics
        trace_lengths = df.groupby('case_id').size()
        
        # Activity distribution entropy
        activity_counts = df['activity'].value_counts(normalize=True)
        entropy = stats.entropy(activity_counts)
        
        # Temporal statistics
        remaining_time_stats = df.groupby('case_id')['remaining_time'].first()
        
        stats_list.append({
            'Dataset': name,
            'Cases': n_cases,
            'Events': n_events,
            'Activities': n_activities,
            'Avg_Trace_Length': trace_lengths.mean(),
            'Std_Trace_Length': trace_lengths.std(),
            'Min_Trace_Length': trace_lengths.min(),
            'Max_Trace_Length': trace_lengths.max(),
            'Activity_Entropy': entropy,
            'Avg_Duration_Days': remaining_time_stats.mean(),
            'Std_Duration_Days': remaining_time_stats.std(),
        })
    
    return pd.DataFrame(stats_list)

def plot_distributional_comparisons(datasets_dict, output_dir='analysis_output'):
    """Create comprehensive visualization of distributional differences"""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Dataset Distributional Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # 1. Trace length distributions
    ax = axes[0, 0]
    for (name, df), color in zip(datasets_dict.items(), colors):
        trace_lengths = df.groupby('case_id').size()
        ax.hist(trace_lengths, bins=50, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('Trace Length (# events)')
    ax.set_ylabel('Density')
    ax.set_title('Trace Length Distribution')
    ax.legend()
    ax.set_xlim(0, 100)
    
    # 2. Activity frequency (log scale)
    ax = axes[0, 1]
    for (name, df), color in zip(datasets_dict.items(), colors):
        activity_counts = df['activity'].value_counts(normalize=True).sort_values(ascending=False)
        ax.plot(range(len(activity_counts)), activity_counts.values, 
                marker='o', label=name, color=color, alpha=0.7)
    ax.set_xlabel('Activity Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Activity Frequency Distribution (Zipf)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Case duration distributions
    ax = axes[0, 2]
    for (name, df), color in zip(datasets_dict.items(), colors):
        durations = df.groupby('case_id')['remaining_time'].first()
        ax.hist(durations, bins=50, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('Case Duration (days)')
    ax.set_ylabel('Density')
    ax.set_title('Case Duration Distribution')
    ax.legend()
    ax.set_xlim(0, 100)
    
    # 4. Vocabulary size comparison
    ax = axes[1, 0]
    vocab_sizes = [df['activity'].nunique() for df in datasets_dict.values()]
    ax.bar(datasets_dict.keys(), vocab_sizes, color=colors)
    ax.set_ylabel('Number of Unique Activities')
    ax.set_title('Vocabulary Size Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Box plot of trace lengths
    ax = axes[1, 1]
    data_for_box = []
    labels_for_box = []
    for name, df in datasets_dict.items():
        trace_lengths = df.groupby('case_id').size()
        data_for_box.append(trace_lengths)
        labels_for_box.append(name)
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Trace Length')
    ax.set_title('Trace Length Distribution (Box Plot)')
    ax.tick_params(axis='x', rotation=45)
    
    # 6. Temporal features - hour of day
    ax = axes[1, 2]
    for (name, df), color in zip(datasets_dict.items(), colors):
        if 'hour_of_day' in df.columns:
            hour_dist = df['hour_of_day'].value_counts(normalize=True).sort_index()
            ax.plot(hour_dist.index, hour_dist.values, marker='o', 
                   label=name, color=color, alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Frequency')
    ax.set_title('Temporal Pattern: Hour of Day')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Activity entropy comparison
    ax = axes[2, 0]
    entropies = []
    for df in datasets_dict.values():
        activity_counts = df['activity'].value_counts(normalize=True)
        entropy = stats.entropy(activity_counts)
        entropies.append(entropy)
    ax.bar(datasets_dict.keys(), entropies, color=colors)
    ax.set_ylabel('Shannon Entropy')
    ax.set_title('Activity Distribution Entropy')
    ax.tick_params(axis='x', rotation=45)
    
    # 8. Cumulative activity coverage
    ax = axes[2, 1]
    for (name, df), color in zip(datasets_dict.items(), colors):
        activity_counts = df['activity'].value_counts(normalize=True).sort_values(ascending=False)
        cumsum = activity_counts.cumsum()
        ax.plot(range(len(cumsum)), cumsum.values, marker='o', 
               label=name, color=color, alpha=0.7)
    ax.set_xlabel('Number of Activities')
    ax.set_ylabel('Cumulative Coverage')
    ax.set_title('Activity Coverage (How many activities for 80%?)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Day of week patterns
    ax = axes[2, 2]
    for (name, df), color in zip(datasets_dict.items(), colors):
        if 'day_of_week' in df.columns:
            dow_dist = df['day_of_week'].value_counts(normalize=True).sort_index()
            ax.plot(dow_dist.index, dow_dist.values, marker='o', 
                   label=name, color=color, alpha=0.7)
    ax.set_xlabel('Day of Week (0=Monday)')
    ax.set_ylabel('Frequency')
    ax.set_title('Temporal Pattern: Day of Week')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributional_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_dir}/distributional_analysis.png")
    plt.close()

def compute_distribution_shifts(datasets_dict):
    """Compute statistical tests for distribution shifts"""
    results = []
    dataset_names = list(datasets_dict.keys())
    
    for i, name1 in enumerate(dataset_names):
        for name2 in dataset_names[i+1:]:
            df1, df2 = datasets_dict[name1], datasets_dict[name2]
            
            # KS test on trace lengths
            trace_len1 = df1.groupby('case_id').size()
            trace_len2 = df2.groupby('case_id').size()
            ks_stat, ks_pval = stats.ks_2samp(trace_len1, trace_len2)
            
            # Chi-square test on activity distributions
            # Sample equal number of activities from each
            min_size = min(len(df1), len(df2))
            act1 = df1['activity'].sample(min_size, replace=False)
            act2 = df2['activity'].sample(min_size, replace=False)
            
            # Get common activities
            common_acts = set(act1.unique()) & set(act2.unique())
            if len(common_acts) > 1:
                act1_filtered = act1[act1.isin(common_acts)]
                act2_filtered = act2[act2.isin(common_acts)]
                
                # Create contingency table
                obs1 = act1_filtered.value_counts()
                obs2 = act2_filtered.value_counts()
                
                # Align indices
                all_acts = sorted(common_acts)
                obs1 = obs1.reindex(all_acts, fill_value=0)
                obs2 = obs2.reindex(all_acts, fill_value=0)
                
                chi2, chi2_pval = stats.chisquare(obs1, obs2)
            else:
                chi2, chi2_pval = np.nan, np.nan
            
            results.append({
                'Dataset_Pair': f"{name1} vs {name2}",
                'KS_Statistic': ks_stat,
                'KS_P_Value': ks_pval,
                'Chi2_Statistic': chi2,
                'Chi2_P_Value': chi2_pval,
                'Significant_Shift': 'Yes' if ks_pval < 0.05 else 'No'
            })
    
    return pd.DataFrame(results)

def main():
    print("Loading datasets...")
    
    # Load all datasets
    datasets = {
        'BPI12': load_and_preprocess(BPI12, 'BPI12'),
        'BPI17': load_and_preprocess(BPI17, 'BPI17'),
        'BPI20PTC': load_and_preprocess(BPI20PrepaidTravelCosts, 'BPI20PTC'),
        'BPI20TPD': load_and_preprocess(BPI20TravelPermitData, 'BPI20TPD'),
        'BPI20RfP': load_and_preprocess(BPI20RequestForPayment, 'BPI20RfP'),
    }
    
    print("\nComputing dataset statistics...")
    stats_df = compute_dataset_statistics(datasets)
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(stats_df.to_string(index=False))
    stats_df.to_csv('analysis_output/dataset_statistics.csv', index=False)
    
    print("\nComputing distribution shifts...")
    shifts_df = compute_distribution_shifts(datasets)
    print("\n" + "="*80)
    print("DISTRIBUTIONAL SHIFTS (KS Test & Chi-Square)")
    print("="*80)
    print(shifts_df.to_string(index=False))
    shifts_df.to_csv('analysis_output/distribution_shifts.csv', index=False)
    
    print("\nGenerating visualizations...")
    plot_distributional_comparisons(datasets)
    
    print("\n" + "="*80)
    print("Analysis complete! Check 'analysis_output/' folder for results.")
    print("="*80)

if __name__ == "__main__":
    main()