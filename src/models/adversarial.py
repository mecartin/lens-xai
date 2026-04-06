import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def fgsm_attack(model, data, target, epsilon):
    """
    Fast Gradient Sign Method (FGSM) attack.
    Creates adversarial examples by perturbing the inputs in the direction of the gradient of the loss.
    """
    if epsilon == 0:
        return data

    # Ensure data requires gradients
    data = data.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(data)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    
    # Calculate loss
    loss = F.cross_entropy(logits, target)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Collect the gradients of the input data
    data_grad = data.grad.data
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data + epsilon * data_grad.sign()
    
    # Clip to maintain valid feature range depending on the dataset.
    # Since features were standardized, there isn't a strict [0, 1] bound.
    # But clipping avoids extreme runaway values.
    # We will just return the perturbed data.
    return perturbed_data

def pgd_attack(model, data, target, epsilon, alpha=0.01, iters=40):
    """
    Projected Gradient Descent (PGD) attack.
    An iterative and stronger version of FGSM.
    """
    if epsilon == 0:
        return data

    original_data = data.clone().detach()
    # Random uniform initialization
    perturbed_data = data.clone().detach() + torch.empty_like(data).uniform_(-epsilon, epsilon)
    perturbed_data.requires_grad = True

    for _ in range(iters):
        outputs = model(perturbed_data)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        loss = F.cross_entropy(logits, target)
        
        model.zero_grad()
        loss.backward()
        
        # Ascent step
        adv_data = perturbed_data + alpha * perturbed_data.grad.sign()
        
        # Projection step (clip the perturbation to be within epsilon of original data)
        eta = torch.clamp(adv_data - original_data, min=-epsilon, max=epsilon)
        perturbed_data = (original_data + eta).clone().detach()
        perturbed_data.requires_grad = True

    return perturbed_data

class AdversarialTrainer:
    """
    Handles robust training using adversarial examples to defend against evasion attacks.
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def train_epoch_adversarial(self, train_loader, optimizer, epsilon=0.05, attack_type="fgsm"):
        """
        Trains for one epoch mixing clean and adversarial data.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # 1. Clean data forward pass
            optimizer.zero_grad()
            outputs_clean = self.model(data)
            logits_clean = outputs_clean[0] if isinstance(outputs_clean, tuple) else outputs_clean
            loss_clean = criterion(logits_clean, target)
            
            # 2. Generate Adversarial Data
            if attack_type == "fgsm":
                data_adv = fgsm_attack(self.model, data, target, epsilon)
            elif attack_type == "pgd":
                data_adv = pgd_attack(self.model, data, target, epsilon, alpha=epsilon/4, iters=10)
            else:
                data_adv = data # No attack
                
            # 3. Adversarial data forward pass
            # We must zero_grad again because generating the attack may have accumulated gradients
            optimizer.zero_grad() 
            outputs_adv = self.model(data_adv)
            logits_adv = outputs_adv[0] if isinstance(outputs_adv, tuple) else outputs_adv
            loss_adv = criterion(logits_adv, target)
            
            # Combine losses (50% clean, 50% adversarial)
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track accuracy on clean data for the epoch metric
            _, predicted = logits_clean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        acc = 100. * correct / total
        return total_loss / len(train_loader), acc

    def evaluate_robustness(self, test_loader, epsilons=[0.0, 0.05, 0.1, 0.2], attack_type="fgsm"):
        """
        Evaluates the model's accuracy on clean data and adversarial examples at different perturbation strengths.
        Returns a dictionary mapping epsilon to accuracy.
        """
        self.model.eval()
        results = {}
        
        for eps in epsilons:
            correct = 0
            total = 0
            
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Generate Attack
                if eps > 0:
                    if attack_type == "fgsm":
                        data_eval = fgsm_attack(self.model, data, target, eps)
                    else:
                        data_eval = pgd_attack(self.model, data, target, eps, alpha=eps/4, iters=10)
                else:
                    data_eval = data
                    
                # Evaluate
                with torch.no_grad():
                    outputs = self.model(data_eval)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    _, predicted = logits.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
            acc = 100. * correct / total
            results[eps] = acc
            print(f"Robustness Eval -> Attack: {attack_type.upper()}, Epsilon: {eps:.2f}, Accuracy: {acc:.2f}%")
            
        return results
