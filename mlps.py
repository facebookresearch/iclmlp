# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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



def create_mlp(hidden_dims, activation_str, use_batch_norm=False, use_layer_norm=False, dropout_rate=0.0):
    layers = []
    
    # Define a dictionary to map activation strings to their corresponding functions
    activation_mapping = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        # Add more activations here as needed
    }
    
    # Get the actual activation function from the mapping
    activation = activation_mapping[activation_str]

    for i, (in_dim, out_dim) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        
        if use_batch_norm and i < len(hidden_dims) - 2:
            layers.append(nn.BatchNorm1d(out_dim))
        
        if use_layer_norm and i < len(hidden_dims) - 2:
            layers.append(nn.LayerNorm(out_dim))
        
        if i < len(hidden_dims) - 2:  # Only add activation if it's not the last pair of dimensions
            layers.append(activation())
        
        if dropout_rate > 0 and i < len(hidden_dims) - 2:
            layers.append(nn.Dropout(dropout_rate))
    
    return nn.Sequential(*layers)


class MLPset(nn.Module):
    def __init__(self, psi_mlp, rho_mlp, xi_mlp):
        super(MLPset, self).__init__()
        self.psi = psi_mlp
        self.rho = rho_mlp
        self.xi = xi_mlp

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Apply psi MLP to each element in the sequence
        psi_outputs = self.psi(x.view(-1, input_dim)).view(batch_size, seq_len, -1)

        # Compute the cumulative sum of psi_outputs along the sequence dimension
        psi_cumsum = torch.cumsum(psi_outputs, dim=1)

        # Compute the mean up to ith point along the sequence dimension
        seq_indices = torch.arange(1, seq_len + 1, dtype=torch.float32, device=psi_outputs.device).view(1, -1, 1)
        psi_mean = psi_cumsum / seq_indices

        # Pass the mean through rho MLP
        rho_output = self.rho(psi_mean.view(-1, psi_outputs.size(-1))).view(batch_size, seq_len, -1)

        # Concatenate rho_output with x[:, 1:, :-1] along the last dimension
        concat_input = torch.cat((rho_output[:, :-1], x[:, 1:, :-1]), dim=-1)

        # Pass the concatenated input through xi MLP
        output = self.xi(concat_input.view(-1, concat_input.size(-1))).view(batch_size, seq_len - 1, -1)
        
        return output

