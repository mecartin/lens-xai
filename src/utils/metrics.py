import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

def classification_report_extended(y_true, y_pred, y_prob=None, class_names=None):
    """
    Computes standard classification report + Cohen's Kappa + AUC (if y_prob provided).
    """
    # Standard Scikit-Learn Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Cohen's Kappa (Agreement against chance)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "kappa": kappa
    }
    
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary AUC
                # y_prob should be shape (N, 2), we extract probability for class 1
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc = roc_auc_score(y_true, prob_pos)
            else:
                # Multi-class AUC OvR
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics["roc_auc"] = auc
        except Exception as e:
            print(f"Warning: Could not compute AUC. {e}")
            metrics["roc_auc"] = None
            
    return metrics, report

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="conf_matrix.png"):
    """
    Saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Saved Confusion Matrix to {save_path}")

def cross_dataset_summary(results_dict):
    """
    Prints a summarized table of results across multiple datasets.
    results_dict format: {"DatasetName": {"accuracy": 95.2, "f1": 94.8, "auc": 98.1}}
    """
    print("\\n" + "="*60)
    print(f"{'Dataset':<15} | {'Accuracy':<10} | {'Macro F1':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    
    for ds_name, metrics in results_dict.items():
        acc = f"{metrics.get('accuracy', 0)*100:.2f}%"
        f1 = f"{metrics.get('macro_f1', 0)*100:.2f}%"
        
        auc_raw = metrics.get('roc_auc')
        auc = f"{auc_raw:.4f}" if auc_raw is not None else "N/A"
        
        print(f"{ds_name:<15} | {acc:<10} | {f1:<10} | {auc:<10}")
    print("="*60 + "\\n")
    
    # Return as DataFrame for easy exporting
    return pd.DataFrame.from_dict(results_dict, orient='index')
