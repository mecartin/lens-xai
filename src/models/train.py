import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path

# Define Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

sys.path.append(str(PROJECT_ROOT))
from src.models.networks import TeacherModel, StudentModel

def kd_loss_function(student_logits, teacher_logits, labels, alpha=0.5, temperature=3.0):
    """
    Knowledge Distillation Loss.
    Combines standard Cross-Entropy Loss with Kullback-Leibler Divergence (KLDiv).
    
    alpha: Weight for CE loss (1-alpha is for KD loss).
    temperature: Smoothes the teacher's logits.
    """
    # 1. Standard Cross Entropy Loss against Ground Truth
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # 2. KL Divergence Loss against Teacher's Soft Labels
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    
    # Combine losses
    total_loss = alpha * ce_loss + (1.0 - alpha) * kd_loss
    return total_loss

def load_data(dataset_name="nsl-kdd", batch_size=256):
    """Loads processed dataset into PyTorch DataLoaders."""
    processed_dir = DATA_DIR / "processed" / dataset_name
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dataset {dataset_name} not found. Run make_dataset.py first.")
    
    X_train_np = np.load(processed_dir / "X_train.npy")
    y_train_np = np.load(processed_dir / "y_train.npy")
    X_test_np = np.load(processed_dir / "X_test.npy")
    y_test_np = np.load(processed_dir / "y_test.npy")
    
    # Sanitize: replace NaN and Inf values with 0 (can occur in some datasets)
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)
        
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    return train_loader, test_loader, input_dim, num_classes

def train_teacher(model, train_loader, device, epochs=5):
    """Trains the Teacher model including H-VAE and deep classifier."""
    print("\\n--- Training Teacher Model ---")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits, weighted_x, att_w, mu1, logvar1, mu2, logvar2 = model(data)
            
            # Combine Classification Loss + KL Divergences from H-VAE
            cls_loss = criterion(logits, target)
            kl_loss_1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
            kl_loss_2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
            
            # Total Loss
            loss = cls_loss + 0.001 * (kl_loss_1 + kl_loss_2)  # Weighted KL loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        acc = 100. * correct / total
        print(f"Teacher Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
    return model

def train_student_kd(teacher, student, train_loader, device, epochs=5, alpha=0.5, temp=3.0):
    """Trains the Student model using Knowledge Distillation."""
    print(f"\\n--- Training Student Model (KD) | Alpha={alpha}, Temp={temp} ---")
    teacher.eval() # Teacher is frozen during distillation
    student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward passes
            with torch.no_grad():
                # Teacher only returns logits as spatial/latent states aren't used for basic KD
                teacher_logits = teacher(data)[0] 
                
            student_logits = student(data)
            
            # Compute KD Loss
            loss = kd_loss_function(student_logits, teacher_logits, target, alpha=alpha, temperature=temp)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        acc = 100. * correct / total
        print(f"Student Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
    return student

def evaluate(model, test_loader, device, name="Model"):
    """Evaluate a trained model."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Handle teacher multi-output vs student single-output
            outputs = model(data)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    acc = 100. * correct / total
    print(f"[{name}] Test Accuracy: {acc:.2f}%")
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_name = "nsl-kdd"
    print(f"Loading {dataset_name}...")
    train_loader, test_loader, input_dim, num_classes = load_data(dataset_name)
    
    # Initialize Models
    teacher = TeacherModel(input_dim=input_dim, num_classes=num_classes).to(device)
    student = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
    
    print(f"Teacher Parameters: {sum(p.numel() for p in teacher.parameters())}")
    print(f"Student Parameters: {sum(p.numel() for p in student.parameters())}")
    
    # 1. Train Teacher
    teacher = train_teacher(teacher, train_loader, device, epochs=5)
    
    # 2. Train Student via KD
    student = train_student_kd(teacher, student, train_loader, device, epochs=5)
    
    # 3. Evaluate Both
    print("\\n--- Final Evaluation ---")
    evaluate(teacher, test_loader, device, name="Teacher")
    evaluate(student, test_loader, device, name="Student (KD)")
    
if __name__ == "__main__":
    main()
