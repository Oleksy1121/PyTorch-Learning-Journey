"""
model_builder.py

Module containing a TinyVGG model architecture based on 
CNN Explainer project: https://poloclub.github.io/cnn-explainer/

Provides:
- TinyVGG: A small CNN for image classification tasks.
"""

from torch import nn


class TinyVGG(nn.Module):
    """
    TinyVGG architecture recreated from CNN Explainer.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        hidden_units (int): Number of units in the convolutional layers.
        out_features (int): Number of output classes.
    """

    def __init__(self, in_channels: int, hidden_units: int, out_features: int):
        super().__init__()

        self.layer_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer_stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=out_features))

    def forward(self, x):
        return self.classifier(self.layer_stack_2(self.layer_stack_1(x)))

