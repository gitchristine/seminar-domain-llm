#!/usr/bin/env python3
"""
Extract experimental results from SLURM log files and create Table 2.
"""

import re
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# Dataset name mapping
DATASET_MAP = {
    'BPI12': 'BPI12',
    'BPI17': 'BPI17',
    'BPI20PrepaidTravelCosts': 'BPI20PTC',
    'BPI20TravelPermitData': 'BPI20TPD',
    'BPI20RequestForPayment': 'BPI20RfP',
    'BPITrafficFines': 'BPITrafficFines'
}

# Backbone name mapping
BACKBONE_MAP = {
    'qwen25-05b': 'Qwen2.5-0.5b',
    'llama32-1b': 'Llama3.2-1b',
    'pm-gpt2': 'PM-GPT2',
    'rnn': 'RNN'
}


def extract_results_from_log(log_file: Path) -> Optional[Dict]:
    """Extract experimental results from a single log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None
    
    # Check if job completed successfully
    if 'completed successfully' not in content:
        if 'Traceback' in content or 'Error' in content or 'error:' in content:
            return {'status': 'ERROR', 'file': log_file.name}
        else:
            return {'status': 'NO OUTPUT', 'file': log_file.name}
    
    result = {'status': 'SUCCESS', 'file': log_file.name}
    
    # Extract dataset
    dataset_match = re.search(r"'log': '(\w+)'", content)
    if dataset_match:
        result['Dataset'] = DATASET_MAP.get(dataset_match.group(1), dataset_match.group(1))
    else:
        result['Dataset'] = 'Unknown'
    
    # Extract backbone
    backbone_match = re.search(r"'backbone': '([\w\-]+)'", content)
    if backbone_match:
        backbone = BACKBONE_MAP.get(backbone_match.group(1), backbone_match.group(1))
        result['Backbone'] = backbone
    else:
        result['Backbone'] = 'Unknown'
    
    # Extract fine-tuning method
    ft_match = re.search(r"'fine_tuning': '(\w+)'", content)
    if ft_match and ft_match.group(1):
        ft_method = ft_match.group(1).upper() if ft_match.group(1) == 'lora' else 'Freezing'
        
        # Check for specific freezing configuration
        freeze_match = re.search(r"'freeze_layers': \[([-\d, ]+)\]", content)
        if freeze_match and ft_method == 'Freezing':
            freeze_layers = freeze_match.group(1).strip()
            result['Backbone'] = f"{result['Backbone']} [Freezing-{freeze_layers}]"
        elif ft_method == 'LORA':
            result['Backbone'] = f"{result['Backbone']} [LoRA]"
        else:
            result['Backbone'] = f"{result['Backbone']} [Freezing]"
    elif ft_match and ft_match.group(1) is None:
        result['Backbone'] = f"{result['Backbone']} [Full]"
    
    # Extract best metrics across all epochs
    # Pattern: Epoch X: ... test_next_activity_acc: 0.XXXX ... test_next_remaining_time_loss: X.XXXX
    epoch_pattern = r'Epoch \d+:.*?test_next_activity_acc: ([\d.]+).*?test_next_remaining_time_loss: ([\d.]+)'
    epochs = re.findall(epoch_pattern, content, re.DOTALL)
    
    if epochs:
        # Find best NA accuracy (highest)
        best_na_acc = max(float(acc) for acc, _ in epochs)
        # Find best RT MSE (lowest)
        best_rt_mse = min(float(mse) for _, mse in epochs)
        
        result['NA Acc.'] = best_na_acc
        result['RT MSE'] = best_rt_mse
    else:
        result['NA Acc.'] = None
        result['RT MSE'] = None
    
    
    # Extract runtime from job statistics
    runtime_match = re.search(r'Job Wall-clock time: (\d+):(\d+):(\d+)', content)
    if runtime_match:
        hours = int(runtime_match.group(1))
        minutes = int(runtime_match.group(2))
        seconds = int(runtime_match.group(3))
        result['Runtime (h)'] = round(hours + minutes/60 + seconds/3600, 3)
    else:
        result['Runtime (h)'] = None
    
    return result



def main():
    # Add the path of the folder where the logs are stored
    logs_dir = Path('logs/qwen_3711941')
    
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        print("Please update the logs_dir path in the script or run from the correct directory.")
        return
    
    # Find all output log files
    log_files = sorted(logs_dir.glob('qwen_*.out'))
    
    if not log_files:
        print(f"No log files found in {logs_dir}/")
        print("Looking for files matching pattern: qwen_*.out")
        return
    
    print(f"Found {len(log_files)} log files\n")
    
    # Extract results from all logs
    results = []
    errors = []
    no_output = []
    
    for log_file in log_files:
        result = extract_results_from_log(log_file)
        if result:
            if result['status'] == 'ERROR':
                errors.append(result)
            elif result['status'] == 'NO OUTPUT':
                no_output.append(result)
            else:
                results.append(result)
    
    # Print summary
    print("="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    print(f"Successful extractions: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"No output: {len(no_output)}")
    print()
    
    # Print errors
    if errors:
        print("="*80)
        print("ERRORS")
        print("="*80)
        for err in errors:
            print(f"  ✗ {err['file']}")
        print()
    
    # Print no output files
    if no_output:
        print("="*80)
        print("NO OUTPUT (Still Running or Failed to Start)")
        print("="*80)
        for no_out in no_output:
            print(f"  ⏳ {no_out['file']}")
        print()
    
    # Create results table
    if results:
        print("="*80)
        print("REPLICATION RESULTS - TABLE 2")
        print("="*80)
        print()
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        # Select and order columns like paper's Table 2
        columns = ['Dataset', 'Backbone', 'NA Acc.', 'RT MSE', 'Runtime (h)']
        
        # Keep only these columns
        df_table = df_results[columns].copy()
        
        # Sort by dataset, then backbone
        df_table = df_table.sort_values(['Dataset', 'Backbone'])
        
        # Format the table for display
        print(df_table.to_string(index=False))
        print()
        
        # Save to CSV -- change the name as needed
        output_csv = Path('results/qwen_replication_results.csv')
        df_table.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv.absolute()}")
        
        
        df_formatted = df_table.copy()
        
        output_formatted = Path('qwen_replication_results_formatted.csv')
        df_formatted.to_csv(output_formatted, index=False)
        print(f"Formatted results saved to: {output_formatted.absolute()}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total experiments: {len(df_table)}")
        print(f"Unique datasets: {df_table['Dataset'].nunique()}")
        print(f"Unique backbones: {df_table['Backbone'].nunique()}")
        print(f"\nAverage NA Acc: {df_table['NA Acc.'].mean():.4f}")
        print(f"Average RT MSE: {df_table['RT MSE'].mean():.4f}")
        print(f"Total runtime: {df_table['Runtime (h)'].sum():.2f} hours")
        
    else:
        print("No successful results found!")


if __name__ == '__main__':
    main()