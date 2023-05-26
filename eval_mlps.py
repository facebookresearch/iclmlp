import os
import torch
import pandas as pd
import numpy as np
import random
from PIL import Image
import argparse
import torch.utils.data as data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import yaml
from quinine import QuinineArgumentParser
import pandas as pd
from schema import schema 
from mlps import MLPset
from Linear_regression_prompts_data import Dataset_generate, linear_regression_prompt_data
from train_mlps import train_erm, seed_worker

## build model 



def eval_model(model, data_input):
    data_new = Dataset_generate(data_input)
    data_loader = data.DataLoader(data_new, batch_size=128)
    model.eval() 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss   = 0.0
    num    = 0 
    with torch.no_grad(): 
        for data_inputs, data_labels in data_loader:
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            loss_eval_module = nn.MSELoss()
            loss_t = torch.square(preds-data_labels)
            
            loss += loss_t.mean(axis=0)
            num  += 1
    # print(f"loss of the model: {loss}%")
    return loss/num


def eval_saved_models(dataset_test, model, argsmodel, run_id, checkpoint_dir):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_checkpoints = 5
    loss_vals_dict = {}

    checkpoint_dir = os.path.join(checkpoint_dir, f'run_{run_id}')

    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_{i * argsmodel.num_epochs // num_checkpoints}.pt')

        # model = MLPset(argsmodel.input_dim, argsmodel.psi_hidden_dim, argsmodel.rho_hidden_dim, argsmodel.psi_output_dim, argsmodel.rho_output_dim, argsmodel.xi_hidden_dim, argsmodel.output_dim)
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)

        loss_vals = eval_model(model, dataset_test)
        checkpoint_index = i * argsmodel.num_epochs // num_checkpoints
        loss_vals_dict[checkpoint_index] = loss_vals.view(-1).tolist()

    return loss_vals_dict

