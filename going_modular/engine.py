"""
engine.py

Module containing functions for training and testing PyTorch models.

Provides:
- train_step: One step of training through the dataloader.
- test_step: One step of evaluation through the dataloader.
- print_metrics: Nicely formatted printing of training/test metrics.
- train: Full training loop across multiple epochs.
"""

import torch
from torch import nn
from typing import Dict, List
from .utils import save_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn,
               optimizer: torch.optim,
               device=device):
    """
    Runs a single training step over a dataloader.

    Args:
        model: A PyTorch model.
        dataloader: Dataloader containing the training data.
        loss_fn: Loss function to compute the loss.
        optimizer: Optimizer to update model weights.
        device: Device to run the computations on ('cpu' or 'cuda').

    Returns:
        Tuple of (train_loss, train_accuracy).
    """

    # set train mode
    model.train()

    # init metrics
    train_loss = 0
    train_accuracy = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # pred
        y_logits = model(X)
        y_pred = y_logits.argmax(dim=1)

        # metrics
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        train_accuracy += ((y_pred == y).sum()/len(y)).item()

        # gradient operations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)

    return train_loss, train_accuracy


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn,
              device=device):
    """
    Runs a single testing step over a dataloader.

    Args:
        model: A PyTorch model.
        dataloader: Dataloader containing the test data.
        loss_fn: Loss function to compute the loss.
        device: Device to run the computations on ('cpu' or 'cuda').

    Returns:
        Tuple of (test_loss, test_accuracy).
    """

    # set eval mode
    model.eval()

    # init metrics
    test_loss = 0
    test_accuracy = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # pred
            y_logits = model(X)
            y_pred = y_logits.argmax(dim=1)

            # metrics
            test_loss += loss_fn(y_logits, y).item()
            test_accuracy += ((y_pred == y).sum()/len(y)).item()

        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

    return test_loss, test_accuracy


def print_metrics(epoch: int, results: Dict[str, float]):
    """
    Nicely formats and prints training and testing metrics per epoch.

    Args:
        epoch: Current epoch number.
        results: Dictionary containing lists of train/test losses and accuracies.
    """

    train_output = ' | '.join("{}: {}".format(k.split('_', 1)[1].capitalize(), round(v[-1], 3 if k=='train_loss' else 2)) for k, v in results.items() if k.split('_', 1)[0] == 'train')
    test_output = ' | '.join("{}: {}".format(k.split('_', 1)[1].capitalize(), round(v[-1], 3 if k=='test_loss' else 2)) for k, v in results.items() if k.split('_', 1)[0] == 'test')

    print(50*'-')
    print(f'Epoch: {epoch}')
    print('Train -> ', train_output)
    print('Test  -> ', test_output)
    print()


def train(model: torch.nn.Module, epochs: int, train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn, optimizer: 
          torch.optim, model_dir_path: str, device=device) -> Dict[str, List]:
    """
    Full training loop for a PyTorch model.

    Tracks and saves the best model based on test loss, and saves the last model.

    Args:
        model: A PyTorch model to train.
        epochs: Number of epochs to train for.
        train_dataloader: Dataloader for the training data.
        test_dataloader: Dataloader for the testing data.
        loss_fn: Loss function.
        optimizer: Optimizer for updating model weights.
        model_dir_path: Directory for saving model results
        device: Device to run the computations on ('cpu' or 'cuda').

    Returns:
        Dictionary containing training and testing metrics.
    """
    
    results = {'train_loss': [],
               'train_accuracy': [],
               'test_loss': [],
               'test_accuracy': []}
    
    best_model_loss = None
    
    for epoch in range(epochs):
        train_loss, train_accuracy = train_step(model,
                                                train_dataloader,
                                                loss_fn,
                                                optimizer,
                                                device)
        
        test_loss, test_accuracy = test_step(model,
                                             test_dataloader,
                                             loss_fn,
                                             device)
        
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_accuracy)
        
        print_metrics(epoch, results)

        if epoch == 0 or test_loss < best_model_loss:
            best_model_loss = test_loss
            save_model(model=model, model_dir_path=model_dir_path, filename='best.pt')

    save_model(model=model, model_dir_path=model_dir_path, filename = 'last.pt')
                
    return results
