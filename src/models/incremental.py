import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy

class IncrementalLearner:
    """
    Handles Incremental Learning using Elastic Weight Consolidation (EWC) 
    and a small Experience Replay buffer to mitigate catastrophic forgetting.
    """
    def __init__(self, model, device="cpu", lambda_ewc=1000, replay_buffer_size=1000):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # EWC state
        self.fisher_matrix = {}
        self.prev_params = {}
        
        # Experience Replay state
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer_x = None
        self.replay_buffer_y = None

    def _compute_fisher_matrix(self, data_loader):
        """
        Estimates the Fisher Information Matrix (FIM) for the model parameters
        using the empirical Fisher (Expected squared gradients).
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p.data) for n, p in self.model.named_parameters() if p.requires_grad}
        
        num_samples = 0
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            
            outputs = self.model(data)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Use log_softmax for Fisher computation
            log_probs = F.log_softmax(logits, dim=1)
            
            # The empirical Fisher uses the gradients of the true label log probabilities
            loss = F.nll_loss(log_probs, target, reduction='sum')
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
                    
            num_samples += data.size(0)
            
        # Average over the dataset
        for n in fisher:
            fisher[n] /= num_samples
            # Normalize to avoid extreme explosion
            fisher[n] = torch.clamp(fisher[n], max=1e2)
            
        return fisher

    def consolidate(self, data_loader):
        """
        Called after training on Task A (old task).
        Computes Fisher matrix and stores current parameters to penalize future changes.
        Also updates the replay buffer with some samples from this old task.
        """
        print("\\n[*] Consolidating Knowledge (EWC Fisher Matrix Computation)...")
        # 1. Update Fisher Matrix
        new_fisher = self._compute_fisher_matrix(data_loader)
        
        if not self.fisher_matrix:
            self.fisher_matrix = new_fisher
        else:
            # Accumulate Fisher from previous tasks
            for n in self.fisher_matrix:
                self.fisher_matrix[n] += new_fisher[n]
                
        # 2. Store current parameters as the reference point for the old task
        self.prev_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
        # 3. Update Replay Buffer (simple random sampling)
        print(f"[*] Updating Experience Replay Buffer (Capacity: {self.replay_buffer_size})...")
        all_x, all_y = [], []
        for x, y in data_loader:
            all_x.append(x)
            all_y.append(y)
        
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        
        # Random subset
        indices = torch.randperm(all_x.size(0))[:self.replay_buffer_size]
        new_replay_x = all_x[indices]
        new_replay_y = all_y[indices]
        
        if self.replay_buffer_x is None:
            self.replay_buffer_x = new_replay_x
            self.replay_buffer_y = new_replay_y
        else:
            # Append and maintain capacity
            self.replay_buffer_x = torch.cat([self.replay_buffer_x, new_replay_x], dim=0)
            self.replay_buffer_y = torch.cat([self.replay_buffer_y, new_replay_y], dim=0)
            
            if self.replay_buffer_x.size(0) > self.replay_buffer_size:
                final_indices = torch.randperm(self.replay_buffer_x.size(0))[:self.replay_buffer_size]
                self.replay_buffer_x = self.replay_buffer_x[final_indices]
                self.replay_buffer_y = self.replay_buffer_y[final_indices]

    def _ewc_loss(self):
        """
        Calculates the Elastic Weight Consolidation penalty.
        L_ewc = (lambda/2) * sum_i( Fisher_i * (param_i - prev_param_i)^2 )
        """
        loss = 0
        if not self.fisher_matrix:
            return loss
            
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher_matrix:
                fisher_val = self.fisher_matrix[n].to(self.device)
                prev_val = self.prev_params[n].to(self.device)
                loss += torch.sum(fisher_val * (p - prev_val) ** 2)
                
        return loss * (self.lambda_ewc / 2)

    def train_new_task(self, new_task_loader, epochs=5, lr=0.001):
        """
        Trains the model on a new task while preserving knowledge of the consolidated old tasks.
        Mixes new task data with replay buffer data.
        """
        print(f"\\n--- Training on New Task (Incremental Learning) ---")
        
        # Create a combined loader if we have a replay buffer
        if self.replay_buffer_x is not None:
            print(f"[*] Fetching {self.replay_buffer_x.size(0)} old task samples from Replay Buffer")
            # Extract new task data
            new_x, new_y = [], []
            for x, y in new_task_loader:
                new_x.append(x)
                new_y.append(y)
            new_x = torch.cat(new_x, dim=0)
            new_y = torch.cat(new_y, dim=0)
            
            # Combine
            combined_x = torch.cat([new_x, self.replay_buffer_x], dim=0)
            combined_y = torch.cat([new_y, self.replay_buffer_y], dim=0)
            
            train_loader = DataLoader(TensorDataset(combined_x, combined_y), batch_size=256, shuffle=True)
            print(f"[*] Combined Dataset Size: {len(combined_y)} samples")
        else:
            train_loader = new_task_loader

        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(data)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Main Classification Loss
                cls_loss = criterion(logits, target)
                
                # Incremental EWC Regularization Loss
                ewc_loss = self._ewc_loss()
                
                loss = cls_loss + ewc_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            acc = 100. * correct / total
            print(f"Incremental Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/len(train_loader):.4f} | EWC Penalty: {ewc_loss.item():.4f} | Acc: {acc:.2f}%")
            
        return self.model
