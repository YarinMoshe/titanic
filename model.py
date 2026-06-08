import torch
import torch.nn as nn

class Mish(nn.Module):
    # Mish activation: x * tanh(ln(1 + e^x)) - smooth non-linear activation
    def forward(self, x): return x * torch.tanh(nn.functional.softplus(x))

class TitanicMLP(nn.Module):
    # Simple MLP
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.LayerNorm(64), 
            Mish(), 
            nn.Dropout(0.3),
            nn.Linear(64, 32), 
            nn.LayerNorm(32), 
            Mish(), 
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Forward pass returning raw logit
        return self.net(x).squeeze(-1)

