import os
import json
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn

config = {
    "block_size": 4,
    "vocab_size": 27,
    "hidden_sizes": [256, 256, 256],
    "n_emb": 64,
    "dropout": 0.2
}

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config["block_size"]
        self.vocab_size = config["vocab_size"]
        self.n_emb = config["n_emb"]
        self.hidden_sizes = config["hidden_sizes"]
        self.dropout_prob = config.get("dropout", 0.0)

        self.E = nn.Embedding(self.vocab_size, self.n_emb)
        self.flat = nn.Flatten()

        layers = []
        input_size = self.n_emb * self.block_size
        for hs in self.hidden_sizes:
          layers.append(nn.Linear(input_size, hs))
          layers.append(nn.ReLU())
          if self.dropout_prob > 0:
            layers.append(nn.Dropout(self.dropout_prob))

          input_size = hs

        layers.append(nn.Linear(input_size, self.vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        X = self.E(X)      # (B, 3, emb)
        X = self.flat(X)   # flatten
        logits = self.net(X)
        return logits


model_path = "model_state.pth"
block_size = 4

model = MLP(config)
model.load_state_dict(torch.load(model_path, map_location="cpu"))

@torch.no_grad()
def generate_name(model, max_len=20):
    context = [0] * block_size
    name = []

    for _ in range(max_len):
        x = torch.tensor([context], dtype=torch.long)
        logits = model(x)
        probs = F.softmax(logits, dim=-1)

        ix = torch.multinomial(probs, num_samples=1).item()  
        
        if ix == 0:
            break

        name.append(ix)
        context = context[1:] + [ix]

    return name

if __name__ == "__main__":
    for _ in range(10):
        print(generate_name(model))