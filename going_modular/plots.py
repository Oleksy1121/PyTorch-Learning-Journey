import torch
from PIL import Image
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torchvision import transforms

def plot_transform(path: str,
                   transform: transforms.Compose or List[transforms.Compose],
                   n: int = 10,
                   fig_width: int = 10,
                   show_img_shapes: bool = True):
    '''
    This function generate subplots with original and transformated images.
    We can plot few transformation. Each transformation are plot in their column.

    Args:
        path: train or test directory path e.g. "data_/your_dataset/train"
        transform: 
        n: Number of random images (Number of subplot rows)
        size_mul: plot size multiplication in inches.
        show_img_shapes: hide/unhide image shape info
    '''

    if type(transform) is transforms.Compose:
        print('Convert compose to List')
        transform = [transform]

    elif type(transform) is list:
        print('list')

    else:
        error_msg = f'''
        Argument "transforms:" got wrong type.
        
        Expected:
            "torchvision.reansforms.Compose"
            or
            "[torchvision.reansforms.Compose]"

        Got:
            {type(transform)}
        
        '''
        raise TypeError(error_msg)

    
    if n > 10:
        n = 10
        print('Cant display more than 10 images')
        
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
    '''
    Funtion for plotting images with predictions. Can plot up to
    60 images from dataset.

    Funtion prepared for PyTorh models for image classification.

    Args:
        model: PyTorch trained model (or untrained)
        dataset: Your data turned in "torch.utils.data.Dataset"
        n_samples: No. of samples (due to optimalisation can
                   be different)
        fig_width: Plot width
    '''
    
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
