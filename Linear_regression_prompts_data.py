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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

## data
class Dataset_generate(data.Dataset):
    def __init__(self,  data):
        
        super().__init__()
        prompt, label    = data
        self.size = prompt.size()[0]
        self.data = prompt
        self.label = label
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label      = self.label[idx]
        return data_point, label



def linear_regression_prompt_data(num_examples,data_dim, prompt_max_len, seed, noise):
    set_seed(seed)
    X = torch.randn(num_examples,data_dim)

    num_tasks = num_examples//prompt_max_len

    prompts = []
    prompts_mat = []

    for i in range(num_tasks):
        
        w = torch.randn(data_dim)
        
        y = X[i*prompt_max_len:(i+1)*prompt_max_len]@w 

        if (noise == True):

             y = y + torch.randn_like(y)
        
        prompts.append((X[i*prompt_max_len:(i+1)*prompt_max_len],y))
        
        y_unsqueezed = y.unsqueeze(1)
        
        prompts_mat.append(torch.cat((X[i*prompt_max_len:(i+1)*prompt_max_len],y_unsqueezed), axis=1))
        
    prompts_tensor     = torch.stack(prompts_mat, dim=0)
    label              = prompts_tensor[:, 1:, -1:]

    # print (prompts_tensor.size())
    # print (label.size())
    return (prompts_tensor, label)

