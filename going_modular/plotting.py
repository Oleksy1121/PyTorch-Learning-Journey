import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path
import random
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from .path_file import models_dir, results_filename
from .predictions import predict
import seaborn as sns

def plot_transform(path: str,
                   transform: transforms.Compose or List[transforms.Compose],
                   n: int = 10,
                   fig_width: int = 10,
                   show_img_shapes: bool = True):
    """
    Plots original and transformed versions of random images from a dataset.

    Args:
        path (str): Path to the dataset directory.
        transform (Compose or list of Compose): Transformations to apply.
        n (int): Number of images to display (max 10).
        fig_width (int): Width of the plot.
        show_img_shapes (bool): Whether to display image shapes.

    Raises:
        TypeError: If `transform` has an invalid type.
    """

    if type(transform) is transforms.Compose:
        transform = [transform]

    elif type(transform) is list:
        pass

    else:
         raise TypeError(
            f'Expected transform to be torchvision.transforms.Compose or List[...] but got {type(transform)}'
         )

    
    if n > 10:
        n = 10
        print("Can\'t display more than 10 images. Showing 10.")
        
    list_of_images = list(Path(path).glob('*/*.jpg'))
    random_images = random.choices(list_of_images, k=n)

    nrows = n
    ncols = len(transform)+1

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_width * nrows / ncols)
    
    for i, img_path in enumerate(random_images):
        
        img = Image.open(img_path)
        img_arr = np.asarray(img)
        label = img_path.parent.stem
        
        ax[i, 0].imshow(img_arr)
        ax[i, 0].set(xticks=[], yticks=[])
        if show_img_shapes:
            ax[i, 0].set_xlabel(list(img_arr.shape), fontsize=10)
        ax[i, 0].set_ylabel(label, fontsize=16)
        ax[0, 0].set_title('Original', fontsize=16)

        
        for j, t in enumerate(transform):
            img_trans = t(img)
            ax[i, j+1].imshow(img_trans.permute(1, 2, 0))
            ax[i, j+1].set(xticks=[], yticks=[])
            if show_img_shapes:
                ax[i, j+1].set_xlabel(list(img_trans.shape), fontsize=10)
            ax[0, j+1].set_title(f'Transform {j+1}', fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_predict_images(model: torch.nn.Module,
                        dataset: torch.utils.data.Dataset,
                        n_samples: int = 10,
                        fig_width: int = 10):
    """
    Plots random images from a dataset with true and predicted labels.

    Args:
        model (nn.Module): Trained (or untrained) PyTorch model.
        dataset (Dataset): Dataset containing images and labels.
        n_samples (int): Number of images to plot (max 60).
        fig_width (int): Width of the entire figure.
    """
    
    if n_samples > 60:
        print("Can't plot more imaages than 60.\nPlotting 60 images.\n")
        n_samples = 60

    cols = 5 if n_samples > 19 else 4 if n_samples > 10 else 3 if n_samples > 4 else 2 if n_samples > 1 else 1
    rows = int(np.ceil(n_samples/cols))
    
    random_idx = random.sample(range(len(dataset)), rows*cols)

    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_width * rows / cols)
    ax = np.array(ax).flatten()

    for i, idx in enumerate(random_idx):
        img, label = dataset[idx]
        
        model.eval()
        with torch.inference_mode():
            y_pred_probs = torch.softmax(model(img.unsqueeze(0)), dim=1)
            y_pred = y_pred_probs.argmax(dim=1)
            
        true_class = dataset.classes[label]
        pred_class = dataset.classes[y_pred]
        
        ax[i].imshow(img.permute(1, 2, 0))
        ax[i].text(0.5, 1.1, true_class + ' |',
                   transform=ax[i].transAxes,
                   ha='right', va='top',
                   fontsize=12, color='black')
        ax[i].text(0.5, 1.1, pred_class,
                   transform=ax[i].transAxes,
                   ha='left', va='top',
                   fontsize=12,
                   color='green' if pred_class == true_class else 'red')
        ax[i].set(xticks=[], yticks=[])


    print(f'Images plot: {cols*rows}\n')
    plt.tight_layout()
    plt.show()


def plot_all_models_result(directory: str = models_dir, result_filename: str = results_filename):
    '''
    Finds result files in model subdirectories and plots specified metrics.

    Searches for `result_filename` in subdirectories within `directory`, reads
    the data (assuming CSV format), and plots each column as a separate series.

    Args:
        directory (str): The main directory to search for model results.
                         Defaults to `models_dir`.
        result_filename (str): The name of the result file to find.
                               Defaults to `results_filename`.
    '''
    
    list_of_model_result = list(Path(directory).glob(f'*/*{result_filename}'))
        
    if list_of_model_result:
        fig = plt.figure(figsize=(12, 15))
    else:
        print(f'Not found "{result_filename}" in "{directory}" directory')

    for result in list_of_model_result:
        df_result = pd.read_csv(result, index_col=0)

        for i, metric in enumerate(df_result.columns):
            plt.subplot(len(df_result.columns), 2, i+1)
            plt.plot(df_result.index, df_result[metric], label=result.parent.stem)
            plt.title(metric)
            plt.legend()


def plot_confussion_matrix(model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           normalize='pred'):
    '''
    Calculates and plots a confusion matrix for a given model and dataset.

    Makes predictions using the model, computes the confusion matrix against
    true labels, and visualizes it using seaborn.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataset (torch.utils.data.Dataset): The dataset for evaluation,
                                             requires `.targets` and `.classes`.
        normalize (str, optional): Normalization method ('true', 'pred', 'all', or None).
                                   Defaults to None.
    '''
    
    y = dataset.targets
    y_pred = predict(model, dataset)

    if not normalize:
        normalize = None
        
    cm = confusion_matrix(y, y_pred, normalize=normalize)

    fmt = '.1%'  if normalize else '.0f'
    title = 'Confusion matrix - normalized' if normalize else 'Confusion matrix'
    
    sns.heatmap(cm, annot=True, cmap='Blues', fmt=fmt, xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title(title)
    plt.xlabel('Prediction')
    plt.ylabel('True')
