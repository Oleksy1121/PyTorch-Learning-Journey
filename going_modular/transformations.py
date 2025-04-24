"""
transformations.py

Module containing predefined sets of torchvision transformations 
for creating PyTorch datasets.
"""

from torchvision import transforms

simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

trivial_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
    ])

auto_augment_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.AutoAugment(),
    transforms.ToTensor()
    ])
