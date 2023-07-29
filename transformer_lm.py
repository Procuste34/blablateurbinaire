import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import wandb

from data_handler import build_data, get_batch, get_voc_size
from model import Transformer_LM

N = 2000
train_split = 0.9
eval_interval = 500
eval_iter = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

def objective(config, wandb_log):
    # train un model avec les HP config

    lr = 10**config['log_learning_rate']
    batch_size = config['batch_size']
    n_layers = config['n_layers']
    d_model = config['d_model']
    n_heads = config['n_heads']
    act = config['act']
    optimizer_hp = config['optimizer']

    build_data('villes.txt', train_split=train_split)

    model = Transformer_LM(n_layers, d_model, n_heads, get_voc_size(), 50, act)
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
        Xb, Yb = get_batch('train', batch_size)

        logits, loss = model(Xb, Yb)

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
                        Xb, Yb = get_batch(split, batch_size)
                        logits, loss = model(Xb, Yb)

                        loss_mean += loss.item()
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
            Xb, Yb = get_batch('val', batch_size)
            logits, loss = model(Xb, Yb)

            val_loss_mean += loss.item()
        val_loss_mean /= eval_iter

    if wandb_log:
        wandb.log({"training_throughput": num_examples_processed/(end_time-start_time)})
        wandb.log({"params_num": sum([p.numel() for p in model.parameters()])})

    return val_loss_mean

def run():
    config = {
               "log_learning_rate": np.log(0.03),
               "batch_size": 1024,
               "n_layers": 2,
               "d_model": 64,
               "n_heads": 2,
               "act": 'selu',
               "optimizer": "AdamW",
               "architecture": "Transformer"
           }

    wandb.init(project="blablateurbinaire", config=config)

    _ = objective(config, wandb_log=True)
    
    wandb.finish()

def run_one_sweep():
    wandb.init(project='blablateurbinaire')
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
            'log_learning_rate': {'min': np.log10(0.0001), 'max': np.log10(0.1)},
            'batch_size': {'values': [256, 512, 1024]},
            'n_layers' : {'values': [1, 2, 4]},
            'd_model' : {'values': [32, 64, 128]},
            'n_heads' : {'values': [1, 2, 4, 8]},
            'act' : {'values': ['selu', 'gelu']},
            'optimizer': {'values': ['AdamW']},
            'architecture': {'values': ['Transformer']}
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='blablateurbinaire')
    wandb.agent(sweep_id, function=run_one_sweep)

#run()
sweep()

