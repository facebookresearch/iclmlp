from collections import OrderedDict
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
from eval import get_run_metrics, read_run_dir, get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler
import numpy as np
import torch.nn as nn
import wandb
import tempfile
from sklearn.linear_model import Ridge
from mlps import create_mlp, MLPset
from baselines_ols_ridge import create_predictors, generate_predictions, create_predictors_ridge, generate_predictions_ridge
from schema_comp import schema 
from quinine import QuinineArgumentParser

def data_prep(xs,ys):
    ys = ys.unsqueeze(-1)
    prompts_tensor     = torch.cat((xs,ys),dim=-1)
    label              = prompts_tensor[:, 1:, -1:]
    return (prompts_tensor, label)


def main(args):
    wandb.init(project="in-context-learning", entity=args.wandb_entity)
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')

    run_path = args.path_txformer
    model, conf = get_model_from_run(run_path)
    n_dims = conf.model.n_dims
    # batch_size = conf.training.batch_size
    batch_size = 1000
    data_sampler = get_data_sampler(conf.training.data, n_dims)
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size,
        **conf.training.task_kwargs
    )
    prompt_len_test =50
    task = task_sampler()
    xs = torch.randn(batch_size,prompt_len_test,10)
    ys = task.evaluate(xs)

    with torch.no_grad():
        pred = model(xs, ys)

    metric = task.get_metric()
    loss1 = metric(pred, ys).numpy()
    mean_loss = loss1[:,1:].mean(axis=0)
    std_loss = loss1[:,1:].std(axis=0)/np.sqrt(batch_size)

    plt.plot(np.arange(1,50),mean_loss, lw=2, label="Transformer 1 (Garg et al.)")
    plt.fill_between(np.arange(1,50), mean_loss.reshape(-1) - std_loss.reshape(-1), mean_loss.reshape(-1) + std_loss.reshape(-1),  alpha=.2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    path_mlp = args.path_mlp
    psi_mlp = create_mlp((11,500,500,500,100), 'ReLU', True, False, 0.0)
    rho_mlp = create_mlp((100,500,500,500,100), 'ReLU', True, False, 0.0)
    xi_mlp  = create_mlp((110,500,1), 'ReLU', True, False, 0.0)

    model_MLPs = MLPset(psi_mlp, rho_mlp, xi_mlp)
    model_MLPs.load_state_dict(torch.load(path_mlp))
    model_MLPs.to(device)
    prompt_tensor, label_tensor = data_prep(xs,ys)
    prompt_tensor, label_tensor = prompt_tensor.to(device), label_tensor.to(device)
    model_MLPs.eval()
    pred_MLPs = model_MLPs(prompt_tensor)
    loss_t = torch.square(pred_MLPs-label_tensor)
    loss_t = torch.square(pred_MLPs-label_tensor)

    mean_loss = loss_t.mean(axis=0).cpu().detach().numpy()
    std_loss = loss_t.std(axis=0).cpu().detach().numpy()/np.sqrt(batch_size)

    plt.plot(np.arange(1,50),mean_loss, lw=2, label="MLP-set")
    plt.fill_between(np.arange(1,50), mean_loss.reshape(-1) - std_loss.reshape(-1), mean_loss.reshape(-1) + std_loss.reshape(-1), alpha=.2)


    # Create the set of predictors
    predictors = create_predictors(xs, ys)
    predictions_ls = generate_predictions(xs, predictors)
    loss_ls = torch.square(predictions_ls.to(device)-label_tensor.squeeze(-1))

    mean_loss = loss_ls.mean(axis=0).cpu().detach().numpy()
    std_loss = loss_ls.std(axis=0).cpu().detach().numpy()/np.sqrt(batch_size)


    plt.plot(np.arange(1,50),mean_loss, lw=2, label="OLS")
    plt.fill_between(np.arange(1,50), mean_loss.reshape(-1) - std_loss.reshape(-1), mean_loss.reshape(-1) + std_loss.reshape(-1),  alpha=.2)
    plt.show()

    # Create the set of predictors
    predictors_ridge = create_predictors_ridge(xs, ys)
    predictions_ridge = generate_predictions_ridge(xs, predictors_ridge)
    loss_ridge = torch.square(predictions_ridge.to(device)-label_tensor.squeeze(-1))
    mean_loss = loss_ridge.mean(axis=0).cpu().detach().numpy()
    std_loss = loss_ridge.std(axis=0).cpu().detach().numpy()/np.sqrt(batch_size)


    plt.plot(np.arange(1,50),mean_loss, lw=2, label="Ridge")
    plt.fill_between(np.arange(1,50), mean_loss.reshape(-1) - std_loss.reshape(-1), mean_loss.reshape(-1) + std_loss.reshape(-1),  alpha=.2)
    plt.xlabel("# in-context examples")
    plt.text(0.5, -0.2, '(c)', size=12, ha="center", 
            transform=plt.gca().transAxes)
    plt.ylabel("squared error")
    plt.legend()
    plt.tight_layout()
    plt.show()


    with tempfile.NamedTemporaryFile(suffix=".png") as temp:
        plt.savefig(temp.name, format="png", dpi=300)
        wandb.log({"ICL in MLPs vs Transformer": wandb.Image(temp.name)})

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    print (args)
    main(args)