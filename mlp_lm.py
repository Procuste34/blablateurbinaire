import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import wandb

from data_handler_bengio import build_data, get_batch, get_voc_size
from model import BengioLM

N = 10000
train_split = 0.9
eval_interval = 500
eval_iter = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

def objective(config, wandb_log):
    # train un model avec les HP config
    # : config.keys = ['context_len', 'learning_rate', 'batch_size', 'embed_dim', 'hidden_dim', 'N']

    context_len = config['context_len']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    embed_dim = config['embed_dim']
    hidden_dim = config['hidden_dim']

    build_data('villes.txt', context_len=config['context_len'], train_split=train_split)

    model = BengioLM(get_voc_size(), context_len, embed_dim, hidden_dim)
    model.to(device)

    if wandb_log:
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
        if wandb_log and (update_num % eval_interval == 0):
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

    if wandb_log:
        wandb.log({"training_throughput": num_examples_processed/(end_time-start_time)})
        wandb.log({"params_num": sum([p.numel() for p in model.parameters()])})

def run():
    config = {
               "learning_rate": 0.03,
               "batch_size": 1024,
               "embed_dim": 16,
               "hidden_dim": 100,
               "context_len": 3
           }

    wandb.init(project="bengio_lm", config=config)

    objective(config, True)
    
    wandb.finish()

def sweep():
    return
    
run()


