"""
predictions.py

Module for making predictions with a trained PyTorch model on a given dataset.
"""

import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def predict(model: torch.nn.Module,
            dataset: torch.utils.data.Dataset, 
            device = 'cpu') -> List[int]:
    """
    Makes predictions on a dataset using a trained model.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataset (torch.utils.data.Dataset): Dataset to make predictions on.
        device (torch.device, optional): Device to perform computations on.

    Returns:
        List[int]: List of predicted class indices.
    """

    model.eval()
    list_of_predictions = []
    
    with torch.inference_mode():
        for img, label in dataset:
            img = img.to(device)
            y_pred = model(img.unsqueeze(dim=0)).argmax(dim=1)
            list_of_predictions.append(y_pred.item())
    return list_of_predictions
