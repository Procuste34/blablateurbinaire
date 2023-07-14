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