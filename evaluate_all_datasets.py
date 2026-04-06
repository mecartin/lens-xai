import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import time

from src.models.networks import TeacherModel
from src.models.train import load_data
from src.federated.federated_trainer import run_federated_training
from src.utils.metrics import classification_report_extended, cross_dataset_summary

def main():
    print("==========================================================")
    print(" LENS-XAI Cross-Dataset Evaluation")
    print(" Evaluating Federated Learning Architecture on 4 Benchmarks")
    print("==========================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    datasets = ["nsl-kdd", "edge-iiotset", "ctu-13", "ukm-ids20"]
    cross_results = {}
    
    for dataset_name in datasets:
        print(f"\n==========================================================")
        print(f" [*] PROCESSING: {dataset_name.upper()}")
        print(f"==========================================================")
        try:
            train_loader, test_loader, input_dim, num_classes = load_data(dataset_name, batch_size=512)
            print(f"    Features: {input_dim} | Classes: {num_classes}")
            
            # Since federated training requires some time, we will just do 1 round, 2 clients
            # and a tiny subset of the test set just for proof-of-concept evaluation speed.
            # In a real run, this would be trained normally over 10-20 epochs.
            
            print("    Running 1-Round Federated Fast-Track Training...")
            model = run_federated_training(n_rounds=1, n_clients=2, dataset_name=dataset_name, local_epochs=1, device=device)
            
            print("    Evaluating Model...")
            model.eval()
            all_preds, all_targets, all_probs = [], [], []
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    outputs = model(data)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    probs = torch.softmax(logits, dim=1)
                    _, preds = logits.max(1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy() if num_classes == 2 else probs.cpu().numpy())
                    
                    if len(all_preds) > 5000: # Limit eval for speed
                        break
                        
            metrics, report = classification_report_extended(np.array(all_targets), np.array(all_preds), np.array(all_probs))
            cross_results[dataset_name.upper()] = metrics
            
        except Exception as e:
            print(f"[-] Failed on dataset {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n\n[*] CROSS-DATASET EVALUATION COMPLETE")
    # This will print the markdown-style summary table
    summary_df = cross_dataset_summary(cross_results)
    
if __name__ == "__main__":
    main()
