import pandas as pd
import torch
import argparse
from pathlib import Path
from datetime import datetime
from next_event_prediction import EVENT_LOGS, NUMERICAL_FEATURES, prepare_data, get_model_config
from ppm.datasets import ContinuousTraces
from ppm.datasets.event_logs import EventFeatures, EventLog, EventTargets
from torch.utils.data import DataLoader
from ppm.datasets.utils import continuous
from ppm.models import NextEventPredictor

def train_and_log(training_config):
    """Train model and save results to CSV"""
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare data
    log = EVENT_LOGS[training_config["log"]]()
    train, test = prepare_data(log.dataframe, log.unbiased_split_params)
    
    event_features = EventFeatures(
        categorical=training_config["categorical_features"],
        numerical=training_config["continuous_features"],
    )
    event_targets = EventTargets(
        categorical=training_config["categorical_targets"],
        numerical=training_config["continuous_targets"],
    )
    
    train_log = EventLog(
        dataframe=train, case_id="case_id", features=event_features,
        targets=event_targets, train_split=True, name=training_config["log"],
    )
    
    test_log = EventLog(
        dataframe=test, case_id="case_id", features=event_features,
        targets=event_targets, train_split=False, name=training_config["log"],
        vocabs=train_log.get_vocabs(),
    )
    
    train_dataset = ContinuousTraces(log=train_log, refresh_cache=True, device=training_config["device"])
    test_dataset = ContinuousTraces(log=test_log, refresh_cache=True, device=training_config["device"])
    
    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], 
                             shuffle=False, collate_fn=continuous)
    test_loader = DataLoader(test_dataset, batch_size=training_config["batch_size"], 
                            shuffle=False, collate_fn=continuous)
    
    model_config = get_model_config(train_log, training_config)
    model = NextEventPredictor(**model_config).to(device=training_config["device"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["lr"], 
                                  weight_decay=training_config["weight_decay"])
    
    # Storage for metrics
    results = []
    
    print("=" * 80)
    print("Training with logging enabled")
    print("=" * 80)
    
    # Training loop
    for epoch in range(training_config["epochs"]):
        model.train()
        train_metrics = {'activity_loss': 0, 'activity_correct': 0, 'activity_total': 0, 
                        'rt_loss': 0, 'rt_count': 0}
        
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["grad_clip"])
            optimizer.step()
            
            # Track metrics
            if 'activity_loss' in outputs:
                train_metrics['activity_loss'] += outputs['activity_loss'].item()
            if 'activity_logits' in outputs and 'activity' in batch['targets']:
                preds = outputs['activity_logits'].argmax(dim=-1)
                targets = batch['targets']['activity']
                train_metrics['activity_correct'] += (preds == targets).sum().item()
                train_metrics['activity_total'] += targets.numel()
            if 'remaining_time_loss' in outputs:
                train_metrics['rt_loss'] += outputs['remaining_time_loss'].item()
                train_metrics['rt_count'] += 1
        
        # Validation
        model.eval()
        test_metrics = {'activity_loss': 0, 'activity_correct': 0, 'activity_total': 0,
                       'rt_loss': 0, 'rt_count': 0}
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch)
                
                if 'activity_loss' in outputs:
                    test_metrics['activity_loss'] += outputs['activity_loss'].item()
                if 'activity_logits' in outputs and 'activity' in batch['targets']:
                    preds = outputs['activity_logits'].argmax(dim=-1)
                    targets = batch['targets']['activity']
                    test_metrics['activity_correct'] += (preds == targets).sum().item()
                    test_metrics['activity_total'] += targets.numel()
                if 'remaining_time_loss' in outputs:
                    test_metrics['rt_loss'] += outputs['remaining_time_loss'].item()
                    test_metrics['rt_count'] += 1
        
        # Calculate averages
        train_act_loss = train_metrics['activity_loss'] / len(train_loader)
        train_act_acc = train_metrics['activity_correct'] / max(train_metrics['activity_total'], 1)
        test_act_loss = test_metrics['activity_loss'] / len(test_loader)
        test_act_acc = test_metrics['activity_correct'] / max(test_metrics['activity_total'], 1)
        train_rt_loss = train_metrics['rt_loss'] / max(train_metrics['rt_count'], 1)
        test_rt_loss = test_metrics['rt_loss'] / max(test_metrics['rt_count'], 1)
        
        # Store results
        results.append({
            'epoch': epoch,
            'train_activity_loss': train_act_loss,
            'train_activity_acc': train_act_acc,
            'test_activity_loss': test_act_loss,
            'test_activity_acc': test_act_acc,
            'train_rt_loss': train_rt_loss,
            'test_rt_loss': test_rt_loss,
        })
        
        print(f"Epoch {epoch}: train_act_loss={train_act_loss:.4f} | train_act_acc={train_act_acc:.4f} | "
              f"test_act_loss={test_act_loss:.4f} | test_act_acc={test_act_acc:.4f} | "
              f"train_rt_loss={train_rt_loss:.4f} | test_rt_loss={test_rt_loss:.4f}")
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{training_config['log']}_{training_config['backbone']}_results_{timestamp}.csv"
    filepath = results_dir / filename
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(filepath, index=False)
    print("=" * 80)
    print(f"Results saved to: {filepath}")
    print("=" * 80)
    
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BPI20PrepaidTravelCosts")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--backbone", type=str, default="rnn")
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    
    args = parser.parse_args()
    
    config = {
        "log": args.dataset,
        "device": args.device,
        "backbone": args.backbone,
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": 0.1,
        "grad_clip": 5.0,
        "n_layers": 1,
        "rnn_type": "lstm",
        "categorical_features": ["activity"],
        "continuous_features": NUMERICAL_FEATURES,
        "categorical_targets": ["activity"],
        "continuous_targets": ["remaining_time"],
        "strategy": "concat",
        "fine_tuning": None,
        "r": None,
        "lora_alpha": None,
        "freeze_layers": None,
    }
    
    train_and_log(config)