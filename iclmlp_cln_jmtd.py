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
from mlps import MLPset, create_mlp
from Linear_regression_prompts_data import Dataset_generate, linear_regression_prompt_data
from train_mlps import train_erm, seed_worker
from eval_mlps  import eval_model, eval_saved_models




def main(args, run_id, wandb_run):
    

    input_dim = args.model.input_dim
    psi_hidden_dim = args.model.psi_hidden_dim
    psi_output_dim = args.model.psi_output_dim
    rho_hidden_dim = args.model.rho_hidden_dim
    rho_output_dim = args.model.rho_output_dim
    xi_hidden_dim  = args.model.xi_hidden_dim
    output_dim     = args.model.output_dim

    if (args.model.architecture == "MLP_vvsmall"):
        hidden_dims = [
            (input_dim,  psi_output_dim),
            (psi_output_dim,  rho_output_dim),
            (rho_output_dim + input_dim - 1, xi_hidden_dim,  output_dim)
        ]

    if (args.model.architecture == "MLP_mvsmall"):
        hidden_dims = [
            (input_dim, psi_hidden_dim, psi_output_dim),
            (psi_output_dim , rho_output_dim),
            (rho_output_dim + input_dim - 1, xi_hidden_dim,  output_dim)
        ]

    if (args.model.architecture == "MLP_vsmall"):
        hidden_dims = [
            (input_dim, psi_hidden_dim, psi_output_dim),
            (psi_output_dim, rho_hidden_dim, rho_output_dim),
            (rho_output_dim + input_dim - 1, xi_hidden_dim,  output_dim)
        ]




    if (args.model.architecture == "MLP_small"):
        hidden_dims = [
            (input_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_output_dim),
            (psi_output_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim,  rho_output_dim),
            (rho_output_dim + input_dim - 1, xi_hidden_dim, output_dim)
        ]


    if (args.model.architecture == "MLP_medium"):
        hidden_dims = [
                (input_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_output_dim),
                (psi_output_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_output_dim),
                (rho_output_dim + input_dim - 1, xi_hidden_dim, xi_hidden_dim, xi_hidden_dim, output_dim)
            ]


    if (args.model.architecture == "MLP_large"):
            hidden_dims = [
                (input_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_output_dim),
                (psi_output_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim,rho_hidden_dim, rho_output_dim),
                (rho_output_dim + input_dim - 1, xi_hidden_dim, xi_hidden_dim, xi_hidden_dim,  xi_hidden_dim, xi_hidden_dim, output_dim)
            ]

    if (args.model.architecture == "MLP_xlarge"):
            hidden_dims = [
                (input_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_hidden_dim, psi_output_dim),
                (psi_output_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim, rho_hidden_dim,rho_hidden_dim, rho_hidden_dim, rho_hidden_dim,rho_hidden_dim, rho_output_dim),
                (rho_output_dim + input_dim - 1, xi_hidden_dim, xi_hidden_dim, xi_hidden_dim,  xi_hidden_dim, xi_hidden_dim, xi_hidden_dim, xi_hidden_dim, output_dim)
            ]



    psi_mlp = create_mlp(hidden_dims[0], args.model.activations[0], args.model.batch_norms[0], args.model.layer_norms[0], args.model.dropouts[0])
    rho_mlp = create_mlp(hidden_dims[1], args.model.activations[0], args.model.batch_norms[0], args.model.layer_norms[0], args.model.dropouts[0])
    xi_mlp  = create_mlp(hidden_dims[2], args.model.activations[0], args.model.batch_norms[0], args.model.layer_norms[0], args.model.dropouts[0])
    
    model = MLPset(psi_mlp, rho_mlp, xi_mlp)
    
    print (model)
    dataset_train = linear_regression_prompt_data(args.data.num_examples,args.data.data_dim, args.data.prompt_max_len_train, seed=1, noise=args.data.noise)
    dataset_test  = linear_regression_prompt_data(args.data.num_examples,args.data.data_dim, args.data.prompt_max_len_test, seed=2, noise=args.data.noise) 


    
    train_erm(dataset_train, model, args.model, run_id, 100, args.out_dir)
    model_eval = MLPset(psi_mlp, rho_mlp, xi_mlp)
    loss_vals_dict = eval_saved_models(dataset_test, model_eval, args.model, run_id, args.out_dir)


    for checkpoint_index, loss_vals_list in loss_vals_dict.items():
        print(f"Checkpoint index: {checkpoint_index}, Loss values: {loss_vals_list}")
        num_examples_list = range(1,len(loss_vals_list)+1)
        results = [[x, y] for (x, y) in zip(num_examples_list, loss_vals_list)]
        table_results = wandb.Table(data=results, columns = ["x", "y"])
        line_plot = wandb.plot.line(table_results, "x", "y", title="ICL_epoch"+str(checkpoint_index))
        wandb.log({"in-context-loss-"+ str(checkpoint_index): line_plot})



    


if __name__ == "__main__":
    
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    wandb_run= wandb.init(project="in-context-learning", entity=args.wandb_entity)
    print (args)
    run_id = wandb_run.id
    main(args, run_id, wandb_run)

