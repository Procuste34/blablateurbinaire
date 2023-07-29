import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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
    
    def sample(self, device, int_to_char, char_to_int, prompt='', max_new_tokens=None):
        prompt = self.context_len * '.' + prompt
        init_len = len(prompt)

        while True:
            context = prompt[-self.context_len:]
            context_tokenized = [char_to_int[c] for c in context]

            logits = self.forward(torch.tensor(context_tokenized, device=device))
            probs = F.softmax(logits, dim=1)

            next_token = torch.multinomial(probs, num_samples=1, replacement=True).item()
            next_char = int_to_char[next_token]

            if next_char == '.':
                break

            prompt += next_char

            if len(prompt)-init_len == max_new_tokens:
                break
        
        return prompt[self.context_len:]
    
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_query):
        super().__init__()

        self.d_query = d_query

        self.X_to_query = nn.Linear(d_model, self.d_query, bias=False)
        self.X_to_key = nn.Linear(d_model, self.d_query, bias=False)
        self.X_to_value = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        # X : (B, T, embed_dim), targets: (B, T)

        B, T, _ = X.size()

        Q = self.X_to_query(X) # (B, T, d_query)
        K = self.X_to_key(X) # (B, T, d_key=d_query)
        V = self.X_to_value(X) # (B, T, d_value)

        QK_T = Q @ torch.transpose(K, 1, 2) # (B, T, T)

        mask = torch.tril(torch.ones((T, T), dtype=torch.int32)).bool()
        QK_T[:, ~mask] = -float("inf")

        attention_scores = torch.softmax(QK_T / math.sqrt(self.d_query), dim=2) # (B, T, T)
        attention = attention_scores @ V # (B, T, d_value=d_head)

        return attention
    
class SelfAttentionMultiHead(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.X_to_query = nn.Linear(d_model, self.n_heads*self.d_head, bias=False) # d_query = d_head as in the Transformer paper
        self.X_to_key = nn.Linear(d_model, self.n_heads*self.d_head, bias=False)
        self.X_to_value = nn.Linear(d_model, self.n_heads*self.d_head, bias=False)

    def forward(self, X):
        # X : (B, T, d_model), targets: (B, T)

        B, T, _ = X.size()

        Q = self.X_to_query(X).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, T, d_query)
        K = self.X_to_key(X).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, T, d_key)
        V = self.X_to_value(X).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, T, d_head=d_value)

        QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, T, T)

        mask = torch.tril(torch.ones((T, T), dtype=torch.int32)).bool()
        QK_T[:, :, ~mask] = -float("inf")

        attention_scores = torch.softmax(QK_T / math.sqrt(self.d_head), dim=3) # (B, n_heads, T, T)
        attention = attention_scores @ V # (B, n_heads, T, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, T, n_heads, d_head)
        attention = attention.contiguous().view(B, T, self.n_heads*self.d_head) # n_heads*d_head = d_model

        return attention
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, act):
        super().__init__()

        # SA sublayer
        #self.sa = SelfAttention(d_model, XX)
        self.sa = SelfAttentionMultiHead(d_model, n_heads)
        self.l1 = nn.LayerNorm(d_model)

        # FC sublayer
        self.fc1 = nn.Linear(d_model, 4*d_model)

        if act == 'selu':
            self.act = F.selu # used in GPT-2/3
        else:
            self.act = F.gelu # new

        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.l2 = nn.LayerNorm(d_model)

    def forward(self, X): # (B, T, d_model)
        H = self.l1(X + self.sa(X)) # SA sublayer
        H = self.l2(H + self.fc2(self.act(self.fc1(H)))) # FC sublayer
        return H
    
class Transformer_LM(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, voc_size, max_len, act='selu'):
        super().__init__()

        self.max_len = max_len
        embed_dim = d_model
        
        self.embed = nn.Embedding(voc_size, embed_dim, padding_idx=0)
        self.PE = nn.Parameter(torch.randn(self.max_len, embed_dim)/10)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, act) for _ in range(n_layers)])

        self.lm_head = nn.Linear(d_model, voc_size)

    def forward(self, W, targets=None):
        # W : (B, T), targets: (B, T)

        B, T = W.size()

        X = self.embed(W) # (B, T, embed_dim=d_model)
        X = X + self.PE[:T]

        H = X
        for layer in self.layers:
            H = layer(H) # (B, T, d_model)
        
        logits = self.lm_head(H) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)

        return logits, loss
    
    def sample(self, device, int_to_char, char_to_int, prompt='', max_new_tokens=None):
        init_len = len(prompt)+1

        #convert prompt to (1, T) input tensor W
        W = torch.zeros(size=(1, init_len), dtype=torch.int64, device=device) # <SOS>

        W[0, 0] = 1 # <SOS> token
        for i, ch in enumerate(prompt):
            W[0, i+1] = char_to_int[ch]

        # autoregressive gen
        while True:
            B, T = W.size()
            
            X = self.embed(W)
            X = X + self.PE[:T]

            H = X
            for layer in self.layers:
                H = layer(H) # (B, T, d_model)

            logits = self.lm_head(H) # (1, T, vocab_size)
            probs = F.softmax(logits[:, -1], dim=1)

            next_token = torch.multinomial(probs, num_samples=1, replacement=True).item()
            next_char = int_to_char[next_token]

            if next_char == '<EOS>':
                break
                
            prompt += next_char

            if len(prompt)-init_len+1 == max_new_tokens or T == self.max_len: # (for PE)
                break

            W = torch.cat([W, torch.tensor(next_token, device=device, dtype=torch.int64).view(1, 1)], dim=1)

        return prompt