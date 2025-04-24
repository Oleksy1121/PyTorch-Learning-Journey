"""
utils.py

Utility functions for saving PyTorch models.
"""

import torch
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List
from .path_file import models_dir, model_dir_prefix

def create_model_directory(MODELS_PATH: str = models_dir, PREFIX: str = model_dir_prefix):
    """
    Creates a new directory for a model with a unique name.

    Checks if the main models directory exists and creates it if not.
    Finds existing model directories with the given prefix and a numeric suffix.
    Generates a new directory name by incrementing the highest existing number.

    Args:
        MODELS_PATH: Path to the main directory for models. Defaults to 'models'.
        PREFIX: Prefix for model directory names. Defaults to 'model_'.

    Returns:
        The path to the newly created model directory.
    """

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
        print(f'Directory "{MODELS_PATH}" has been created')
    
    model_dir_list = [x for x in os.listdir(MODELS_PATH) if (x[:len(PREFIX)] == PREFIX)
                      and (x[len(PREFIX):].isdigit())]
    
    if model_dir_list:
        dir_index_list = [int(x[len(PREFIX):]) for x in model_dir_list]
        dir_index_new = str(max(dir_index_list) + 1)
    else: 
        dir_index_new = '0'
     
    dir_name = PREFIX + dir_index_new
    model_dir_path = os.path.join(MODELS_PATH, dir_name)
    
    try:
        os.mkdir(model_dir_path)
        print(f'Directory "{model_dir_path}" has been created')
        
    except FileExistsError:
        print(f'Directory "{model_dir_path}" arleady exist')

    return model_dir_path


def save_model(model: torch.nn.Module,
               model_dir_path: str,
               filename: str = 'model.pt'):
    """
    Saves a PyTorch model's state_dict to the 'models/' directory.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        filename (str): The name of the file to save the model weights to.
                        Must include '.pt' or '.pth' extension.

    Example:
        save_model(model, "model.pt")
    """
    torch.save(model.state_dict(), os.path.join(model_dir_path, filename))
    print(f'Saved model to "{os.path.join(model_dir_path, filename)}"')


def save_results_to_csv(model_results: Dict[str, List[float]],
                        model_dir_path: str,
                        filename: str = 'results.csv'):
    """
    Saves model results to a CSV file in a newly created model directory.

    Args:
        model_results: Dictionary where keys are metric names and values are lists of results.
        filename: Name of the CSV file to save to. Defaults to 'results.csv'.

    """
    try:
        model_results_df = pd.DataFrame(model_results, columns=model_results.keys())
        model_results_df = model_results_df.astype('float')
        file_path = os.path.join(model_dir_path, filename)
        model_results_df.to_csv(file_path)
        print(f'Saved model results to: "{file_path}"')
    except Exception as e:
        print(f"An error occurred while saving results: {e}")
