import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import time

from src.models.networks import TeacherModel, StudentModel
from src.models.train import load_data, kd_loss_function
from src.federated.federated_trainer import run_federated_training
from src.models.adversarial import AdversarialTrainer, fgsm_attack
from src.models.incremental import IncrementalLearner
from src.utils.metrics import classification_report_extended, cross_dataset_summary
from src.models.quantize import quantize_model, compare_model_sizes, benchmark_inference

def run_demo():
    print("==========================================================")
    print(" LENS-XAI Phase 3 & 4 Demonstration")
    print(" Federated Learning, Adversarial Robustness, Incremental")
    print(" Edge Quantization, and Comprehensive XAI Metrics")
    print("==========================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    dataset_name = "nsl-kdd"
    
    print("[*] Loading Distributed Benchmark Data...")
    try:
        train_loader, test_loader, input_dim, num_classes = load_data(dataset_name, batch_size=256)
    except FileNotFoundError:
        print(f"Error: Processed {dataset_name} data not found. Please run 'python src/data/make_dataset.py' first.")
        return

    # 1. Federated Learning
    print("\n==========================================================")
    print(" 1. FEDERATED LEARNING (Privacy-Preserving Aggregation)")
    print("==========================================================")
    global_model = run_federated_training(n_rounds=2, n_clients=3, dataset_name=dataset_name, local_epochs=1, device=device)
    
    # Evaluate global model
    global_model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = global_model(data)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy() if num_classes == 2 else probs.cpu().numpy())
            
            if len(all_preds) > 10000: # Limit for speed
                break
                
    metrics, report = classification_report_extended(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    print(f"Federated Model Macro F1: {metrics['macro_f1']:.4f} | ROC-AUC: {metrics.get('roc_auc', 0):.4f}")

    # 2. Adversarial Robustness
    print("\n==========================================================")
    print(" 2. ADVERSARIAL DEFENSE (FGSM Attack & Robust Training)")
    print("==========================================================")
    adv_trainer = AdversarialTrainer(global_model, device=device)
    
    # Create a small subset to evaluate robustness quickly
    subset_indices = np.random.choice(len(test_loader.dataset), 1000, replace=False)
    eval_loader = DataLoader(Subset(test_loader.dataset, subset_indices), batch_size=256)
    
    print("[*] Evaluating baseline vulnerability...")
    baseline_robustness = adv_trainer.evaluate_robustness(eval_loader, epsilons=[0.0, 0.05, 0.1])
    
    print("\n[*] Training with Adversarial Examples (Defense Formulation)...")
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
    adv_trainer.train_epoch_adversarial(eval_loader, optimizer, epsilon=0.05)
    
    print("\n[*] Re-evaluating robustness after defense...")
    defended_robustness = adv_trainer.evaluate_robustness(eval_loader, epsilons=[0.0, 0.05, 0.1])

    # 3. Incremental Learning (EWC)
    print("\n==========================================================")
    print(" 3. INCREMENTAL LEARNING (Elastic Weight Consolidation)")
    print("==========================================================")
    # Simulate Task A (e.g., specific attack types or just a subset of data)
    task_a_indices = list(range(1000))
    task_b_indices = list(range(1000, 2000))
    
    task_a_loader = DataLoader(Subset(train_loader.dataset, task_a_indices), batch_size=256, shuffle=True)
    task_b_loader = DataLoader(Subset(train_loader.dataset, task_b_indices), batch_size=256, shuffle=True)
    
    incremental_learner = IncrementalLearner(global_model, lambda_ewc=5000, replay_buffer_size=200, device=device)
    
    # Train on Task A (in this case, just fine-tune quickly)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
    for data, target in task_a_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = global_model(data)[0]
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        optimizer.step()
        
    print("[*] Evaluating Task A Baseline Performance:")
    incremental_learner.model.eval()
    correct_a = 0
    with torch.no_grad():
        for d, t in task_a_loader:
            d, t = d.to(device), t.to(device)
            preds = incremental_learner.model(d)[0].max(1)[1]
            correct_a += preds.eq(t).sum().item()
    print(f"    Task A Initial Accuracy: {100.*correct_a/len(task_a_indices):.2f}%")
        
    # Consolidate Task A
    incremental_learner.consolidate(task_a_loader)
    
    # Train on Task B with EWC
    incremental_learner.train_new_task(task_b_loader, epochs=2)
    
    # Re-evaluate Task A
    incremental_learner.model.eval()
    correct_a_retained = 0
    with torch.no_grad():
        for d, t in task_a_loader:
            d, t = d.to(device), t.to(device)
            preds = incremental_learner.model(d)[0].max(1)[1]
            correct_a_retained += preds.eq(t).sum().item()
    print(f"\n[*] Evaluating Task A Retained Performance (EWC + Replay):")
    print(f"    Task A Retained Accuracy: {100.*correct_a_retained/len(task_a_indices):.2f}%")


    # 4. Edge Quantization
    print("\n==========================================================")
    print(" 4. EDGE DEPLOYMENT (INT8 Post-Training Quantization)")
    print("==========================================================")
    print("[*] Creating Lightweight Student Model (KD target)")
    student = StudentModel(input_dim=input_dim, num_classes=num_classes)
    
    # Populate student with some learned weights (simulate KD completion)
    # We could run KD here, but for demo speed we just quantize immediately
    print("[*] Quantizing Student Model...")
    quantized_student = quantize_model(student)
    
    # Compare
    fp32_size, int8_size, red = compare_model_sizes(student, quantized_student)
    
    # Evaluate student on its device, quantized on CPU
    print("\n[*] Benchmarking Latency...")
    print("FP32 Student:")
    benchmark_inference(student, test_loader, num_batches=10, device=device)
    print("INT8 Student (CPU only):")
    benchmark_inference(quantized_student, test_loader, num_batches=10, device="cpu")


    # 5. XAI & Metrics
    print("\n==========================================================")
    print(" 5. XAI INTEPRETABILITY & REPORTING")
    print("==========================================================")
    try:
        from src.utils.xai import SHAPExplainer
        print("[*] Initializing SHAP Explainer using background sample (N=50)...")
        # Get background data
        bg_data = []
        for d, _ in test_loader:
            bg_data.append(d)
            break
        bg_data = torch.cat(bg_data, dim=0)[:50]
        
        # Test instances
        test_samples = bg_data[:5]
        
        feature_names = [f"Feature_{i}" for i in range(input_dim)]
        
        explainer = SHAPExplainer(global_model, bg_data, feature_names=feature_names)
        shap_values = explainer.explain(test_samples, nsamples=100)
        explainer.plot_summary(shap_values, test_samples, save_path="shap_demo_plot.png")
        print("[*] Summary Plot generated successfully.")
    except Exception as e:
        print(f"[-] SHAP evaluation skipped or failed: {e}")
        
    print("\n==========================================================")
    print(" DEMO COMPLETE")
    print("==========================================================\n")

if __name__ == "__main__":
    run_demo()
