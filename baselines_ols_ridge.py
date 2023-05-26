# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import numpy as np

from sklearn.linear_model import Ridge

# # Function to create least squares predictor with bias
def least_squares_predictor(X, y):
    return torch.linalg.lstsq(X, y).solution 

# Function to create a set of predictors for each task
def create_predictors(tasks_xs, tasks_ys):
    num_tasks, num_points, _ = tasks_xs.shape
    predictors = []

    for task_idx in range(num_tasks):
        task_predictors = []
        for i in range(1, num_points):
            X = tasks_xs[task_idx, :i]
            y = tasks_ys[task_idx, :i]
            task_predictors.append(least_squares_predictor(X, y))
        predictors.append(task_predictors)

    return predictors


# Function to generate predictions using the predictors
def generate_predictions(tasks_xs, predictors):

    num_tasks, num_points, _ = tasks_xs.shape
    predictions = []

    for task_idx in range(num_tasks):
        task_predictions = []
        for i in range(num_points - 1):
            X = tasks_xs[task_idx, i+1].unsqueeze(0)
            task_predictions.append(torch.matmul(X, predictors[task_idx][i]).squeeze())
        predictions.append(task_predictions)

    return torch.tensor(predictions)

def ridge_predictor(X, y, alpha=1.0):
    model = Ridge(alpha=alpha,fit_intercept=False)
    model.fit(X, y)
    return torch.tensor(model.coef_), torch.tensor(model.intercept_)

# Function to create a set of predictors for each task
def create_predictors_ridge(tasks_xs, tasks_ys, alpha=1.0):
    num_tasks, num_points, _ = tasks_xs.shape
    predictors = []

    for task_idx in range(num_tasks):
        task_predictors = []
        for i in range(1, num_points):
            X = tasks_xs[task_idx, :i].numpy()
            y = tasks_ys[task_idx, :i].numpy()
            coef, intercept = ridge_predictor(X, y, alpha)
            task_predictors.append((coef, intercept))
        predictors.append(task_predictors)

    return predictors

# Function to generate predictions using the predictors
def generate_predictions_ridge(tasks_xs, predictors):
    num_tasks, num_points, _ = tasks_xs.shape
    predictions = []

    for task_idx in range(num_tasks):
        task_predictions = []
        for i in range(num_points - 1):
            X = tasks_xs[task_idx, i+1].numpy()
            coef, intercept = predictors[task_idx][i]
            pred = torch.tensor(X).matmul(coef) 
            task_predictions.append(pred.squeeze())
        predictions.append(task_predictions)

    return torch.tensor(predictions)

    
