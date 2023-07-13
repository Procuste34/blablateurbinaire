import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import wandb

from data_handler_bengio import build_data, get_batch, get_voc_size

context_len = 3

lr = 0.03
batch_size = 1024
embed_dim = 16
hidden_dim = 100

N = 10000
train_split = 0.9
eval_interval = 500
eval_iter = 50

build_data('villes.txt', context_len=context_len, train_split=train_split)

device = "cuda" if torch.cuda.is_available() else "cpu"

class BengioLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(get_voc_size(), embed_dim)
        self.fc1 = nn.Linear(context_len * embed_dim, hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, get_voc_size())

    def forward(self, x):
        x = self.embed(x).view(-1, context_len*embed_dim)

        z1 = self.fc1(x)
        a1 = F.tanh(z1)

        logits = self.lm_head(a1)

        return logits
    
    def sample(self, prompt="", max_new_tokens=None):
        return
    
wandb.init(project="bengio_lm",
           config={
               "learning_rate": lr,
               "batch_size": batch_size,
               "embed_dim": embed_dim,
               "hidden_dim": hidden_dim,
               "context_len": context_len
           })

model = BengioLM()
model.to(device)

start_time = time.time()
wandb.watch(model, log="all")

for update_num in range(N):
    Xb, Yb = get_batch(batch_size, 'train', device)

    logits = model(Xb)

    loss = F.cross_entropy(logits, Yb)

    for p in model.parameters():
        p.grad = None

    loss.backward()

    for p in model.parameters():
        p.data += -lr * p.grad

    # eval : track loss (train & val), update_to_data
    if update_num % eval_interval == 0:
        to_log = {}

        with torch.no_grad():
            model.eval()
            for split in ['train', 'val']:
                loss_mean = 0
                for i in range(eval_iter):
                    Xb, Yb = get_batch(batch_size, split, device)
                    logits = model(Xb)

                    loss_mean += F.cross_entropy(logits, Yb).item()
                loss_mean /= eval_iter
                to_log["loss_" + split] = loss_mean
            model.train()

            scalars_dict = {}

            for name, p in model.named_parameters():
                scalars_dict[name] = (lr*p.grad.std() / p.data.std()).log10().item()
        
        wandb.log(to_log | {"update_to_data": scalars_dict}, step=update_num)

end_time = time.time()
num_examples_processed = N * batch_size

print("training throughput = {} examples/s".format(str(num_examples_processed/(end_time-start_time))))
wandb.log({"training_throughput": num_examples_processed/(end_time-start_time)})
wandb.log({"params_num": sum([p.numel() for p in model.parameters()])})

wandb.finish()