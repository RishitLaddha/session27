"""
This module implements a simple decoder-only Transformer model using custom layer functions.
It avoids using high-level torch.nn layers for core operations by relying on custom functions
defined in custom_layers.py.

This version uses a small randomly generated text dataset for a quick test run and runs for 5 epochs.
"""

import torch
import torch.optim as optim
import math
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from custom_layers import linear_custom, gelu, layer_norm, softmax

# -----------------------------------------------------------------------------
# Transformer Configuration Data Class
# -----------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    vocab_size: int = 1000   # Vocabulary size
    max_seq_len: int = 64    # Maximum sequence length
    dim: int = 256           # Model dimension
    num_layers: int = 2      # Number of Transformer layers
    num_heads: int = 2       # Number of attention heads
    dropout: float = 0.1     # Dropout rate (not used in this simple version)

# -----------------------------------------------------------------------------
# Custom Transformer Model (Decoder-only)
# -----------------------------------------------------------------------------

class CustomTransformer:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.d_model = config.dim
        
        # Wrap parameters to ensure they are leaf tensors.
        self.embed = torch.nn.Parameter(torch.randn(config.vocab_size, config.dim) * 0.1)
        
        self.Wq = torch.nn.Parameter(torch.randn(config.dim, config.dim) * 0.1)
        self.Wk = torch.nn.Parameter(torch.randn(config.dim, config.dim) * 0.1)
        self.Wv = torch.nn.Parameter(torch.randn(config.dim, config.dim) * 0.1)
        self.Wo = torch.nn.Parameter(torch.randn(config.dim, config.dim) * 0.1)
        
        self.ff_weight1 = torch.nn.Parameter(torch.randn(config.dim, config.dim * 4) * 0.1)
        self.ff_bias1 = torch.nn.Parameter(torch.zeros(config.dim * 4))
        self.ff_weight2 = torch.nn.Parameter(torch.randn(config.dim * 4, config.dim) * 0.1)
        self.ff_bias2 = torch.nn.Parameter(torch.zeros(config.dim))
        
        self.proj_weight = torch.nn.Parameter(torch.randn(config.dim, config.vocab_size) * 0.1)
        self.proj_bias = torch.nn.Parameter(torch.zeros(config.vocab_size))
        
        self.params = [
            self.embed, self.Wq, self.Wk, self.Wv, self.Wo,
            self.ff_weight1, self.ff_bias1, self.ff_weight2, self.ff_bias2,
            self.proj_weight, self.proj_bias
        ]
    
    def attention(self, Q, K, V):
        d = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
        attn = softmax(scores, dim=-1)
        return attn @ V

    def forward(self, x_indices):
        x = self.embed[x_indices]  # (N, seq_len, d_model)
        Q = linear_custom(x, self.Wq, bias=0)
        K = linear_custom(x, self.Wk, bias=0)
        V = linear_custom(x, self.Wv, bias=0)
        attn_out = self.attention(Q, K, V)
        attn_out = linear_custom(attn_out, self.Wo, bias=0)
        x = x + attn_out  # Residual connection
        
        ff = linear_custom(x, self.ff_weight1, self.ff_bias1)
        ff = gelu(ff)
        ff = linear_custom(ff, self.ff_weight2, self.ff_bias2)
        x = x + ff  # Residual connection
        
        logits = linear_custom(x, self.proj_weight, self.proj_bias)
        return logits

    def get_parameters(self):
        return self.params

# -----------------------------------------------------------------------------
# Simple Text Dataset for Quick Test Run
# -----------------------------------------------------------------------------

class SimpleTextDataset(Dataset):
    def __init__(self, vocab_size, seq_len, size=20):
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.targets = torch.roll(self.data, shifts=-1, dims=1)
        self.targets[:, -1] = 0
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# -----------------------------------------------------------------------------
# Training Function for the Transformer
# -----------------------------------------------------------------------------

def train_transformer(model, dataloader, epochs=5, lr=0.001, device='cpu'):
    optimizer = optim.SGD(model.get_parameters(), lr=lr)
    logs = []
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)
        total_loss = 0.0
        total_tokens = 0
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            logits = model.forward(data)
            N, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(N * seq_len, vocab_size)
            targets_flat = targets.view(N * seq_len)
            
            loss = criterion(logits_flat, targets_flat)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item() * (N * seq_len)
            total_tokens += (N * seq_len)
            
            # (Optional: you can print progress per batch if desired)
        avg_loss = total_loss / total_tokens
        log_entry = f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}"
        print("[Transformer]", log_entry)
        logs.append(log_entry)
    return logs




# -----------------------------------------------------------------------------
# Main function for Transformer Training
# -----------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    config = TransformerConfig(vocab_size=1000, max_seq_len=16, dim=128, num_layers=2, num_heads=2)
    model = CustomTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SimpleTextDataset(vocab_size=config.vocab_size, seq_len=16, size=20)  # adjust size as needed
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    print("Training Custom Transformer on toy text data (training for 5 epochs)...")
    transformer_logs = train_transformer(model, dataloader, epochs=5, lr=0.001, device=str(device))
    
    with open("README_Transformer.txt", "w") as f:
        f.write("Custom Transformer Training Logs:\n")
        for log in transformer_logs:
            f.write(log + "\n")

if __name__ == "__main__":
    main()
