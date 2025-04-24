"""
train.py

Main script to train a TinyVGG model on a custom dataset using PyTorch.
"""

import torch
from torchvision import transforms
from time import time
from . import data_setup, model_builder, engine, transformations, utils, path_file
import argparse

def main():

    # setup parser
    parser = argparse.ArgumentParser(description="Train a TinyVGG model on custom dataset.")
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate for optimizer.')
    parser.add_argument('-hu', '--hidden_units', type=int, default=10, help='Number of hidden units in TinyVGG.')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of CPU cores for data loading.')
    parser.add_argument('-t', '--transform', type=str, default='simple_transform', help='Type of train data transform',
                       choices=['simple_transform', 'trivial_transform', 'auto_augment_transform'])
    parser.add_argument('-tt', '--test_transform', type=str, default='simple_transform', help='Type of test data transform',
                       choices=['simple_transform', 'trivial_transform', 'auto_augment_transform'])
    args = parser.parse_args()

    # set train and test directiries
    train_dir = path_file.train_dir
    test_dir = path_file.test_dir
    
    # device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # define transformation
    try:
        transform = getattr(transformations, args.transform)
        test_transform = getattr(transformations, args.test_transform)
    except AttributeError:
        raise ValueError('Invalid transformtion selected')

    # create model directories
    model_dir_path = utils.create_model_directory()

    # create dataloaders
    data = data_setup.DataManager(train_dir=train_dir, test_dir=test_dir, transform=transform)
    
    train_dataloader, test_dataloader, classes = data.create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # init the classification model
    model = model_builder.TinyVGG(in_channels=3,
                                  hidden_units=args.hidden_units,
                                  out_features=len(classes)).to(device)
    
    # set loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate)
    
    # training
    time_start = time()
    model_results = engine.train(model=model,
                                 epochs=args.epochs,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 loss_fn=loss_fn,
                                 optimizer=optimizer,
                                 model_dir_path=model_dir_path,
                                 device=device)

    utils.save_results_to_csv(model_results=model_results, model_dir_path=model_dir_path)
    print(f'Training completed in {(time() - time_start):.0f} seconds')


if __name__ == '__main__':
    main()
