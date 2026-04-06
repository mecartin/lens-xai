import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

class FederatedClient:
    """
    Simulates a single edge node (client) in the Federated Learning network.
    """
    def __init__(self, client_id, data_loader, device="cpu"):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device

    def train(self, global_model, epochs=1, lr=0.001):
        """
        Trains the global model locally to produce a local update.
        """
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                # Teacher returns (logits, weighted_x, att_weights, mu1, logvar1, mu2, logvar2)
                # Or student just returns logits. Handle both.
                outputs = local_model(data)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                loss = criterion(logits, target)
                
                # If it's the Teacher model with H-VAE, ignore KL loss for simplicity in FL client
                # or we can just train the classification head. Let's keep it simple with cls loss.
                loss.backward()
                optimizer.step()
                
        # Return the parameter differences (Delta) or the state dict
        return local_model.state_dict(), len(self.data_loader.dataset)


class FederatedServer:
    """
    Simulates the central aggregation server.
    """
    def __init__(self, global_model):
        self.global_model = global_model

    def add_dp_noise(self, state_dict, sigma=0.01):
        """
        Conceptual Differential Privacy: Adds Gaussian noise to the aggregated weights.
        """
        noisy_state_dict = copy.deepcopy(state_dict)
        for key in noisy_state_dict.keys():
            if "weight" in key or "bias" in key:
                if noisy_state_dict[key].dtype in (torch.float32, torch.float64):
                    noise = torch.randn_like(noisy_state_dict[key]) * sigma
                    noisy_state_dict[key] += noise
        return noisy_state_dict

    def aggregate(self, client_updates, apply_dp=False, dp_sigma=0.01):
        """
        Performs Federated Averaging (FedAvg).
        client_updates: list of tuples (state_dict, num_samples)
        """
        total_samples = sum(num_samples for _, num_samples in client_updates)
        
        # Initialize an empty state dict for the aggregated weights
        aggregated_state_dict = copy.deepcopy(client_updates[0][0])
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = torch.zeros_like(aggregated_state_dict[key]).float()
            
            # Weighted sum over clients
            for client_state, num_samples in client_updates:
                weight = num_samples / total_samples
                # Cast the client parameter to float32 to avoid dtype mismatches (e.g. Integer types like num_batches_tracked)
                aggregated_state_dict[key] += (client_state[key].float() * weight)
                
            # Cast back to original dtype if necessary (e.g., int64 for batch norm tracking)
            if client_updates[0][0][key].dtype != torch.float32:
                 aggregated_state_dict[key] = aggregated_state_dict[key].to(client_updates[0][0][key].dtype)

        if apply_dp:
            aggregated_state_dict = self.add_dp_noise(aggregated_state_dict, sigma=dp_sigma)

        # Update the global model
        self.global_model.load_state_dict(aggregated_state_dict)
        return self.global_model

def evaluate_global(model, test_loader, device):
    """Evaluates the global model on the test set."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total

def run_federated_training(n_rounds=3, n_clients=3, dataset_name="nsl-kdd", local_epochs=1, device=None):
    """
    End-to-end simulation of Federated Learning.
    Note: Requires dataset arrays to have been created via make_dataset.py.
    """
    from src.models.networks import TeacherModel
    from src.models.train import load_data
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\\n--- Starting Federated Learning Simulation ({n_clients} Clients, {n_rounds} Rounds) ---")
    
    # 1. Load Data
    train_loader, test_loader, input_dim, num_classes = load_data(dataset_name, batch_size=256)
    
    # 2. Partition Data among Clients (IID split for simplicity)
    full_dataset = train_loader.dataset
    dataset_size = len(full_dataset)
    indices = np.random.permutation(dataset_size)
    split_size = dataset_size // n_clients
    
    clients = []
    for i in range(n_clients):
        start_idx = i * split_size
        end_idx = dataset_size if i == n_clients - 1 else (i + 1) * split_size
        subset_indices = indices[start_idx:end_idx]
        subset = Subset(full_dataset, subset_indices)
        client_loader = DataLoader(subset, batch_size=256, shuffle=True)
        clients.append(FederatedClient(client_id=i, data_loader=client_loader, device=device))
        
    print(f"Data partitioned: ~{split_size} samples per client.")
    
    # 3. Initialize Global Model and Server
    global_model = TeacherModel(input_dim=input_dim, num_classes=num_classes).to(device)
    server = FederatedServer(global_model)
    
    # Evaluate baseline
    initial_acc = evaluate_global(global_model, test_loader, device)
    print(f"Global Model Initial Acc: {initial_acc:.2f}%")
    
    # 4. Federated Training Loop
    for r in range(n_rounds):
        print(f"\\n[Round {r+1}/{n_rounds}]")
        client_updates = []
        
        # Local Training
        for client in clients:
            state_dict, n_samples = client.train(server.global_model, epochs=local_epochs, lr=0.001)
            client_updates.append((state_dict, n_samples))
            print(f"  - Client {client.client_id} finished training.")
            
        # Global Aggregation with tracking privacy budget
        # We apply a slight DP noise starting from round 2 just as demonstration
        apply_dp = (r > 0) 
        server.aggregate(client_updates, apply_dp=apply_dp, dp_sigma=0.005)
        
        # Evaluate
        round_acc = evaluate_global(server.global_model, test_loader, device)
        dp_status = "(DP Noise Applied)" if apply_dp else ""
        print(f"  -> Global Model Acc {dp_status}: {round_acc:.2f}%")
        
    print("\\n--- Federated Learning Finished ---")
    return server.global_model
