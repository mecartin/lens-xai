import torch
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import shap
except ImportError:
    print("Warning: SHAP is not installed. Run `pip install shap` to use XAI features.")
    
try:
    import lime
    import lime.lime_tabular
except ImportError:
    print("Warning: LIME is not installed. Run `pip install lime` to use XAI features.")

class SHAPExplainer:
    """
    Explainable AI (XAI) using SHAP (SHapley Additive exPlanations)
    Identifies global and local feature importance.
    """
    def __init__(self, model, background_data, feature_names=None):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        
        # We need a wrapper function for SHAP since our model returns multiple outputs
        # and SHAP expects a clean tensor -> tensor function
        def predict_func(x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            
            # If the model is running on GPU, push input to GPU
            device = next(self.model.parameters()).device
            x = x.to(device)
            
            with torch.no_grad():
                outputs = self.model(x)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                return logits

        self.predict_func = predict_func
        
        # DeepExplainer is ideal for PyTorch but can be finicky with complex architectures
        # Use KernelExplainer as a robust fallback for the predict function
        # Convert background to numpy for KernelExplainer
        if isinstance(background_data, torch.Tensor):
            bg_np = background_data.cpu().numpy()
        else:
            bg_np = background_data
            
        print("[*] Initializing SHAP KernelExplainer (this may take a moment)...")
        self.explainer = shap.KernelExplainer(self.predict_func, bg_np)

    def explain(self, test_instances, nsamples=100):
        """
        Generates SHAP values for the given test instances.
        nsamples controls the approximation accuracy of the KernelExplainer.
        """
        if isinstance(test_instances, torch.Tensor):
            test_np = test_instances.cpu().numpy()
        else:
            test_np = test_instances
            
        print(f"[*] Computing SHAP values for {len(test_np)} instances...")
        shap_values = self.explainer.shap_values(test_np, nsamples=nsamples)
        return shap_values

    def plot_summary(self, shap_values, test_instances, save_path="shap_summary.png"):
        """
        Plots the global summary of feature importances.
        """
        if isinstance(test_instances, torch.Tensor):
            test_np = test_instances.cpu().numpy()
        else:
            test_np = test_instances
            
        plt.figure()
        # shap_values from KernelExplainer for multi-class is a list of arrays
        # Plot for class 1 (usually the attack class) if it exists
        vals_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
        
        shap.summary_plot(vals_to_plot, test_np, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved SHAP Summary Plot to {save_path}")


class LIMEExplainer:
    """
    Explainable AI (XAI) using LIME (Local Interpretable Model-agnostic Explanations).
    Provides instance-level (local) interpretability.
    """
    def __init__(self, model, training_data, feature_names=None, class_names=None):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names
        
        if class_names is None:
            class_names = ["Normal", "Attack"]
            
        if isinstance(training_data, torch.Tensor):
            train_np = training_data.cpu().numpy()
        else:
            train_np = training_data

        # Initialize LIME Tabular Explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            train_np,
            feature_names=self.feature_names,
            class_names=class_names,
            discretize_continuous=True
        )
        
        # Predict function wrapper for numpy -> numpy probabilities
        def predict_proba(x_np):
            device = next(self.model.parameters()).device
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = torch.nn.functional.softmax(logits, dim=1)
                return probs.cpu().numpy()

        self.predict_proba = predict_proba

    def explain_instance(self, instance, num_features=10):
        """
        Explains a single instance.
        """
        if isinstance(instance, torch.Tensor):
            inst_np = instance.cpu().numpy().flatten()
        else:
            inst_np = instance.flatten()
            
        exp = self.explainer.explain_instance(
            inst_np, 
            self.predict_proba, 
            num_features=num_features
        )
        return exp

    def plot_explanation(self, exp, save_path="lime_explanation.png"):
        """
        Saves the LIME explanation to an image.
        """
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[*] Saved LIME Explanation to {save_path}")
