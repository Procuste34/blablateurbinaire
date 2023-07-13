import torch
import torch.nn as nn
import torch.nn.functional as F

class BengioLM(nn.Module):
    def __init__(self, voc_size, context_len, embed_dim, hidden_dim):
        super().__init__()

        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(voc_size, embed_dim)
        self.fc1 = nn.Linear(context_len * embed_dim, hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, voc_size)

    def forward(self, x):
        x = self.embed(x).view(-1, self.context_len*self.embed_dim)

        z1 = self.fc1(x)
        a1 = F.tanh(z1)

        logits = self.lm_head(a1)

        return logits
    
    def sample(self, prompt="", max_new_tokens=None):
        return