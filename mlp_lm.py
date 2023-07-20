import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import wandb

from data_handler_bengio import build_data, get_batch, get_voc_size
from model import BengioLM

N = 20000
train_split = 0.9
eval_interval = 500
eval_iter = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

def objective(config, wandb_log):
    # train un model avec les HP config
    # : config.keys = ['context_len', 'log_learning_rate', 'batch_size', 'embed_dim', 'hidden_dim', 'optimizer']

    context_len = config['context_len']
    lr = 10**config['log_learning_rate']
    batch_size = config['batch_size']
    embed_dim = config['embed_dim']
    hidden_dim = config['hidden_dim']
    optimizer_hp = config['optimizer']

    build_data('villes.txt', context_len=config['context_len'], train_split=train_split)

    model = BengioLM(get_voc_size(), context_len, embed_dim, hidden_dim)
    model.to(device)

    if optimizer_hp == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_hp == 'SGD_M':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_hp == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
        
    start_time = time.time()

    if wandb_log:
        wandb.watch(model, log="all")

    for update_num in range(N):
        Xb, Yb = get_batch(batch_size, 'train', device)

        logits = model(Xb)

        loss = F.cross_entropy(logits, Yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    with torch.no_grad():
        val_loss_mean = 0
        for _ in range(eval_iter):
            Xb, Yb = get_batch(batch_size, 'val', device)
            logits = model(Xb)

            val_loss_mean += F.cross_entropy(logits, Yb).item()
        val_loss_mean /= eval_iter

    if wandb_log:
        wandb.log({"training_throughput": num_examples_processed/(end_time-start_time)})
        wandb.log({"params_num": sum([p.numel() for p in model.parameters()])})

    return val_loss_mean

def run():
    config = {
               "log_learning_rate": np.log(0.03),
               "batch_size": 1024,
               "embed_dim": 16,
               "hidden_dim": 100,
               "context_len": 3,
               "optimizer": "Adam",
               "architecture": "Bengio"
           }

    wandb.init(project="communes_lm", config=config)

    _ = objective(config, wandb_log=True)
    
    wandb.finish()

def run_one_sweep():
    wandb.init(project='communes_lm')
    val_loss = objective(wandb.config, wandb_log=False)
    wandb.log({'final_val_loss': val_loss})

def sweep():
    sweep_configuration = {
        'method': 'random',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'final_val_loss'
            },
        'parameters': 
        {
            'log_learning_rate': {'min': np.log10(0.0001), 'max': np.log10(0.3)},
            'batch_size': {'values': [1024]},
            'embed_dim': {'values': [8, 16, 32, 64]},
            'hidden_dim': {'values': [50, 100, 300, 500]},
            'context_len': {'values': [3, 5, 8]},
            'optimizer': {'values': ['SGD', 'SGD_M', 'Adam', 'AdamW']},
            'architecture': {'values': ['Bengio']}
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='communes_lm')
    wandb.agent(sweep_id, function=run_one_sweep)

#run()
sweep()

