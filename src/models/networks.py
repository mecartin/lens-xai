import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionFeatureSelector(nn.Module):
    """
    Attention-Based Feature Selection Module.
    Dynamically weights input features to prioritize informative spatial patterns 
    for rare/evolving attacks before passing to the H-VAE.
    """
    def __init__(self, input_dim):
        super(SelfAttentionFeatureSelector, self).__init__()
        # Simple attention mechanism (Squeeze-and-Excitation style for 1D tabular)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Calculate attention weights
        weights = self.attention(x)
        # Apply weights to input features
        weighted_x = x * weights
        return weighted_x, weights

class HierarchicalVAE(nn.Module):
    """
    Hierarchical Variational Autoencoder (H-VAE).
    Captures multi-scale representations (packet, flow, session).
    Uses multiple latent spaces to model different temporal granularities.
    """
    def __init__(self, input_dim, latent_dim_1=32, latent_dim_2=16):
        super(HierarchicalVAE, self).__init__()
        
        # Level 1 Encoder (e.g., Flow level)
        self.enc1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.fc_mu1 = nn.Linear(128, latent_dim_1)
        self.fc_var1 = nn.Linear(128, latent_dim_1)
        
        # Level 2 Encoder (e.g., Session level, conditioned on Level 1)
        self.enc2 = nn.Sequential(
            nn.Linear(latent_dim_1, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.fc_mu2 = nn.Linear(64, latent_dim_2)
        self.fc_var2 = nn.Linear(64, latent_dim_2)
        
        # Level 2 Decoder
        self.dec2 = nn.Sequential(
            nn.Linear(latent_dim_2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim_1)
        )
        
        # Level 1 Decoder
        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim_1 * 2, 128), # Concat z1 and reconstructed z1'
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode Level 1
        h1 = self.enc1(x)
        mu1, logvar1 = self.fc_mu1(h1), torch.clamp(self.fc_var1(h1), min=-20, max=5)
        z1 = self.reparameterize(mu1, logvar1)
        
        # Encode Level 2
        h2 = self.enc2(z1)
        mu2, logvar2 = self.fc_mu2(h2), torch.clamp(self.fc_var2(h2), min=-20, max=5)
        z2 = self.reparameterize(mu2, logvar2)
        
        # Decode Level 2
        z1_recon = self.dec2(z2)
        
        # Decode Level 1 (using both original z1 and hierarchical reconstruction)
        z_combined = torch.cat([z1, z1_recon], dim=1)
        x_recon = self.dec1(z_combined)
        
        return x_recon, mu1, logvar1, mu2, logvar2, z1, z2

class TeacherModel(nn.Module):
    """
    The Teacher Model: Attention + H-VAE Encoder + Deep Classifier.
    """
    def __init__(self, input_dim, num_classes, latent_dim_1=32, latent_dim_2=16):
        super(TeacherModel, self).__init__()
        self.attention = SelfAttentionFeatureSelector(input_dim)
        self.h_vae = HierarchicalVAE(input_dim, latent_dim_1, latent_dim_2)
        
        # Deep Classifier Head (Takes concatenated latent states z1, z2)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim_1 + latent_dim_2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 1. Attention
        weighted_x, att_weights = self.attention(x)
        
        # 2. H-VAE Encoding
        h1 = self.h_vae.enc1(weighted_x)
        mu1, logvar1 = self.h_vae.fc_mu1(h1), torch.clamp(self.h_vae.fc_var1(h1), min=-20, max=5)
        z1 = self.h_vae.reparameterize(mu1, logvar1)
        
        h2 = self.h_vae.enc2(z1)
        mu2, logvar2 = self.h_vae.fc_mu2(h2), torch.clamp(self.h_vae.fc_var2(h2), min=-20, max=5)
        z2 = self.h_vae.reparameterize(mu2, logvar2)
        
        # 3. Classification
        z_combined = torch.cat([z1, z2], dim=1)
        logits = self.classifier(z_combined)
        
        return logits, weighted_x, att_weights, mu1, logvar1, mu2, logvar2

class StudentModel(nn.Module):
    """
    The Student Model: A highly compressed, lightweight MLP for Edge deployment.
    """
    def __init__(self, input_dim, num_classes):
        super(StudentModel, self).__init__()
        
        # We also include a simplified Attention in student to process raw inputs similarly
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # Shallow Classifier Head (~40-60% smaller than Teacher)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
            # Removed extra deep layers and large latents
        )

    def forward(self, x):
        weights = self.attention(x)
        weighted_x = x * weights
        logits = self.classifier(weighted_x)
        return logits
