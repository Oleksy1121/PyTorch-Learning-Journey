import torch
from torch import nn
import torchvision
from torchinfo import summary
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path
import zipfile
import requests
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from typing import Tuple, List, Dict
from time import time

#%% gathering data
PATH_DATA = Path('C:/Users/Marcin/Documents/Python/Udemy/PyTorch/data')
IMAGE_DATA = PATH_DATA / 'pizza_stak_sushi'


if IMAGE_DATA.exists():
    print('Directory arleady exist')
else:
    Path.mkdir(IMAGE_DATA)
    print('Directory has been created')

with open(IMAGE_DATA / 'pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
    f.write(request.content)
    print('Zip has been downloaded')

with zipfile.ZipFile(IMAGE_DATA / 'pizza_steak_sushi.zip', 'r') as zip_ref:
    zip_ref.extractall(IMAGE_DATA)
    print('Zip has been extracted')

Path.unlink(IMAGE_DATA / 'pizza_steak_sushi.zip')
print('Zip file has been deleted')

#%% lets inspect our directory
def throught_dir(PATH):
    for path, dirname, filename in Path.walk(PATH):
        print(f'There is {len(dirname)} directories, and {len(filename)} in {path}')

throught_dir(IMAGE_DATA)

#%%visualize
image_path_list = list(Path.glob(IMAGE_DATA, '*/*/*.jpg'))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem

img = Image.open(random_image_path)

print(f'Image class: {image_class}')
print(f'Image width: {img.width}')
print(f'Image Height: {img.height}')
img

# matplotlib
img_array = np.asarray(img)
plt.imshow(img_array)
plt.axis(False)
plt.title(f'Image class: {image_class} | shape: {img_array.shape}')
plt.show()

#%%  transformation
data_transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor()    
    ])

img_trans = data_transformation(img)

# lets visualize few samples
samples = 2

fig, ax = plt.subplots(samples, 2)
for i in range(samples):
    img_path = random.choice(image_path_list)
    img_class = img_path.parent.stem
    img = Image.open(img_path)
    img_array = np.asarray(img)
    img_trans = data_transformation(img)
    
    ax[i, 0].imshow(img_array)
    ax[i, 0].axis(False)
    ax[i, 0].set_title(f'{img_class} - Original')
    
    
    ax[i, 1].imshow(img_trans.permute(1, 2, 0))
    ax[i, 1].axis(False)
    ax[i, 1].set_title(f'{img_class} - Transposed')
    

#%% lets turn data to dataset

'''
There is 2 option's to create custom dataset:
    1. Image Folder
    2. Create a custom datase class
'''

#%% 1. create custom dataset via ImageFolder
train_data_path = IMAGE_DATA / 'train'
test_data_path = IMAGE_DATA / 'test'

train_dataset = torchvision.datasets.ImageFolder(
    root=train_data_path,
    transform=data_transformation
    )

test_dataset = torchvision.datasets.ImageFolder(
    root = test_data_path,
    transform=data_transformation
    )

len(train_dataset), len(test_dataset)
classes = train_dataset.classes


# plot random image from dataset
random_img_index = random.randint(0, len(train_dataset))
plt.imshow(train_dataset[random_img_index][0].permute(1, 2, 0))
plt.title(f'Image class: {train_dataset.classes[train_dataset[random_img_index][1]]}')
plt.axis(False)
plt.show()


#%% 2. Custom dataset
# def for get class and dictionary
def get_classes(PATH: int) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(PATH))
    if not classes:
        raise FileNotFoundError(f"Couldn't find classes in {PATH}")
    
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

get_classes(train_data_path)


# custom class
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root_dir = root,
        self.transform = transform
        self.paths = list(root.glob('*/*.jpg'))
        self.classes, self.class_to_idx = get_classes(root)
    
    def load_image(self, index: int):
        'Return PIL image'
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self):
        'Return len of the dataset'
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
        'Return image and class'
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor()    
    ])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()    
    ])


train_dataset_custom = CustomDataSet(train_data_path, train_transform)
test_dataset_custom = CustomDataSet(test_data_path, test_transform)

img, label = train_dataset_custom[0]

train_dataset_custom.classes
train_dataset_custom.class_to_idx

plt.imshow(img.permute(1, 2, 0))
plt.axis(False)
plt.title(f'Image class: {train_dataset_custom.classes[label]}')

#%% compare ImageFolder dataset to our CustomDataSet
(len(train_dataset) == len(train_dataset_custom)) & (len(test_dataset) == len(test_dataset_custom))
train_dataset.classes == train_dataset_custom.classes
test_dataset.class_to_idx == train_dataset_custom.class_to_idx

#%% hepler function to visualize random samples from dataset
def plot_random_images(dataset: torch.utils.data.Dataset,
                       n: int = 10,
                       display_shape=True,
                       seed=None):
    'Its function that plotting random samples from our dataset'
    
    if n > 10:
        n = 10
        print("Number of samples 'n' should be less than 10")
    
    if seed:
        random.seed(seed)
        
    random_indexes = random.sample(range(len(dataset)), k=n)
    
    plt.figure(figsize=(16, 4))
    for i, sample_idx in enumerate(random_indexes):
        img, label = dataset[sample_idx]
        
        plt.subplot(1, n, i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.axis(False)
        title = dataset.classes[label]
        if display_shape:
            title += f'\n{list(img.shape)}'
        
        
        plt.title(title)
    
plot_random_images(train_dataset, 10)
plot_random_images(train_dataset_custom, 10)

#%% lets try transformations with image augumentation

def plot_transform(path, transform, n):
    
    if n > 10:
        n=10
        print('Cant display more than 10 images')
        
    list_of_images = list(path.glob('*/*.jpg'))
    random_images = random.choices(list_of_images, k=n)

    fig, ax = plt.subplots(nrows=n, ncols=2)
    for i, img_path in enumerate(random_images):
        
        img = Image.open(img_path)
        img_arr = np.asarray(img)
        
        img_trans = transform(img)
        
        label = img_path.parent.stem
        
        ax[i, 0].imshow(img_arr)
        ax[i, 0].set_title(f'Org | {label} | {img_arr.shape}')
        ax[i, 0].axis('off')
        
        ax[i, 1].imshow(img_trans.permute(1, 2, 0))
        ax[i, 1].set_title(f'T | {label} | {list(img_trans.shape)}')
        ax[i, 1].axis('off')
    plt.show()
        
    

train_transform_v1 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31),
    torchvision.transforms.ToTensor()
    ])

plot_transform(train_data_path, train_transform_v1, 4)

train_transform_v2 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.AugMix(),
    torchvision.transforms.ToTensor()
    ])

plot_transform(train_data_path, train_transform_v2, 4)

train_transform_v3 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.AutoAugment(),
    torchvision.transforms.ToTensor()
    ])

plot_transform(train_data_path, train_transform_v3, 4)

#%% data loader
BATCH_SIZE = 8

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

len(train_dataloader)
len(test_dataloader)

batch_features, batch_labels = next(iter(train_dataloader))
batch_features.shape, batch_labels.shape

plt.imshow(batch_features[0].permute(1, 2, 0))
plt.axis(False)
plt.title(classes[batch_labels[0].item()])
plt.show()


#%%
'''
Lets build first base model TinyVGG from CNNExplainer

'''

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transformation
simple_transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor()
    ])

# dataset
train_dataset_simple = torchvision.datasets.ImageFolder(train_data_path,
                                                        transform=simple_transformation)
test_dataset_simple = torchvision.datasets.ImageFolder(test_data_path,
                                                       transform=simple_transformation)
                                                        
plot_random_images(train_dataset_simple, 10)


# dataloader
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

train_dataloader_simple = torch.utils.data.DataLoader(train_dataset_simple,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True)

test_dataloader_simple = torch.utils.data.DataLoader(test_dataset_simple,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=False)


#%% class replicated from CNN Explainer -> https://poloclub.github.io/cnn-explainer/

class TinyVGG(nn.Module):
    def __init__(self, in_channels, hidden_units, out_features):
        super().__init__()
        
        self.layer_stack_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.layer_stack_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13, # there is trict to calculat in_features
                      out_features=out_features)
            )
    
    def forward(self, x):
        x = self.layer_stack_1(x)
        #print(x.shape)
        x = self.layer_stack_2(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x)
        return x

#%% init the model
model_0 = TinyVGG(in_channels=3, hidden_units=10, out_features=3).to(device)
model_0.state_dict()

# lets push some values to check classifier layer in features
dummy_x = torch.rand((1, 3, 64, 64))
dummy_x.shape
model_0(dummy_x).shape

#%% train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn,
               optimizer: torch.optim,
               device=device):
    
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

#%% test step
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn,
              device=device):
    
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
        
        test_loss /= batch
        test_accuracy /= batch
    
    return test_loss, test_accuracy
    
#%% print function    
def print_metrics(epoch: int, results: Dict[str, float]):
    
    train_output = ' | '.join("{}: {}".format(k.split('_', 1)[1].capitalize(), round(v[-1], 3 if k=='train_loss' else 2)) for k, v in results.items() if k.split('_', 1)[0] == 'train')
    test_output = ' | '.join("{}: {}".format(k.split('_', 1)[1].capitalize(), round(v[-1], 3 if k=='test_loss' else 2)) for k, v in results.items() if k.split('_', 1)[0] == 'test')
        
    print(50*'-')
    print(f'Epoch: {epoch}')
    print('Train -> ', train_output)
    print('Test  -> ', test_output)
    print()    
    
#%% train loop
def train(model: torch.nn.Module,
          epochs: int,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn,
          optimizer: torch.optim,
          device=device):
    
    results = {'train_loss': [],
               'train_accuracy': [],
               'test_loss': [],
               'test_accuracy': []}
    
    for epoch in range(epochs):
        train_loss, train_accuracy = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_accuracy = test_step(model, test_dataloader, loss_fn, device)
        
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_accuracy)
        
        print_metrics(epoch, results)
                
    return results
#%% training basemodel
model_0 = TinyVGG(in_channels=3, hidden_units=10, out_features=len(train_dataset_simple.classes))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

EPOCHS = 10
model_0_results = train(model_0, EPOCHS, train_dataloader_simple, test_dataloader_simple, loss_fn, optimizer, device)

#%% plotting model_0 results
def plot_results(results):
    plt.figure(figsize=(12, 7))
    
    plt.subplot(1,2,1)
    plt.plot(range(len(results['train_loss'])), results['train_loss'], c='b', label='Train')
    plt.plot(range(len(results['test_loss'])), results['test_loss'], c='r', label='Test')
    plt.legend()
    plt.title('Loss')
    plt.show()
    
    plt.subplot(1,2,2)
    plt.plot(range(len(results['train_accuracy'])), results['train_accuracy'], c='b', label='Train')
    plt.plot(range(len(results['test_accuracy'])), results['test_accuracy'], c='r', label='Test')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

plot_results(model_0_results)

#%%
'''
LETS BUILD MODEL_1 WITH THE AUGUMENTATION APPLIED

'''


#%%
# define trivial augumentation transformation for train data
trivial_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31),
    torchvision.transforms.ToTensor()    
    ])

# create dataset
train_dataset_trivial = torchvision.datasets.ImageFolder(root=train_data_path,
                                                            transform=trivial_transform)
# plots random dataset images
plot_random_images(train_dataset_trivial, n=10)
plot_random_images(test_dataset_simple, n=10)

# define train dataloader
BATCH_SIZE = 32

train_dataloader_trivial = torch.utils.data.DataLoader(dataset=train_dataset_trivial,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True)

#%% train model on augumented samples
# init the model
model_1 = TinyVGG(in_channels=3, hidden_units=10, out_features=len(train_dataset_trivial.classes))

#summary(model_1, input_size=(32, 1, 64, 64))

# set loss funcion and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

# train
EPOCHS = 10
model_1_result = train(model=model_1, 
                       epochs=EPOCHS,
                       train_dataloader=train_dataloader_trivial,
                       test_dataloader=test_dataloader_simple,
                       loss_fn=loss_fn,
                       optimizer=optimizer)

#%% plot the results
plot_results(model_1_result)

#%% compare the models
def plot_compare_models(**kwargs):
    plt.figure(figsize=(12, 6))

    for model_name, results in kwargs.items():
        print(model_name)
        #print(results)
        plt.subplot(2,2,1)
        
        plt.plot(range(len(results['train_loss'])), results['train_loss'], label=model_name)
        plt.title('Train Loss Coprasion')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(2,2,2)
        plt.plot(range(len(results['test_loss'])), results['test_loss'], label=model_name)
        plt.title('Test Loss Coprasion')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(2,2,3)
        plt.plot(range(len(results['train_accuracy'])), results['train_accuracy'], label=model_name)
        plt.title('Train Accuracy Coprasion')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.subplot(2,2,4)
        plt.plot(range(len(results['test_accuracy'])), results['test_accuracy'], label=model_name)
        plt.title('Test Accuracy Coprasion')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
    plt.tight_layout()
    plt.show()
    

plot_compare_models(model_0=model_0_results, model_1=model_1_result)

#%% predict 
def make_prediction(model: nn.Module, dataset: torch.utils.data.Dataset):
    pred_list =[]
    model.eval()
    
    with torch.inference_mode():
        for (i, l) in dataset:
            y_pred = model(i.unsqueeze(0)).argmax(dim=1)
            pred_list.append(y_pred.item())
    
    return pred_list

# plotting random predictions
def plot_random_predictions(model: nn.Module, dataset: torch.utils.data.Dataset, n_samples=10):
    random_idx = random.sample(range(len(dataset)), n_samples)
    
    fig = plt.figure()
    rows = 5 if n_samples > 19 else 4 if n_samples > 10 else 3 if n_samples > 4 else 2 if n_samples>1 else 1
    cols = int(np.ceil(n_samples/rows))
    
    for i, idx in enumerate(random_idx):
        img, label = dataset[idx]
        
        model.eval()
        with torch.inference_mode():
            y_pred_probs = torch.softmax(model(img.unsqueeze(0)), dim=1)
            y_pred = y_pred_probs.argmax(dim=1)
        
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.axis(False)
        plt.title(f'T: {dataset.classes[label]} | P: {dataset.classes[y_pred]}\n{y_pred_probs.max():.1%}', color='green' if label==y_pred else 'red')
    
    plt.tight_layout()
    plt.show()
    
    
y_pred = make_prediction(model_0, test_dataset_simple[0][:])
y_pred = make_prediction(model_0, (test_dataset_simple[10][0], test_dataset_simple[10][1]))

y_pred = plot_random_predictions(model_0, test_dataset_simple, n_samples=10)
y_pred = plot_random_predictions(model_1, test_dataset_simple, n_samples=10)

#%% confussion matrix
model_0_cm = confusion_matrix(test_dataset_simple.targets, make_prediction(model_0, test_dataset_simple))
sns.heatmap(model_0_cm, annot=True, cmap='Blues', xticklabels=test_dataset_simple.classes, yticklabels=test_dataset_simple.classes)

model_1_cm = confusion_matrix(test_dataset_simple.targets, make_prediction(model_1, test_dataset_simple))
sns.heatmap(model_1_cm, annot=True, cmap='Blues', xticklabels=test_dataset_simple.classes, yticklabels=test_dataset_simple.classes)


#%% lets predict custom image
custom_img_path = Path('C:/Users/Marcin/Documents/Python/Udemy/PyTorch/data/pizza_stak_sushi') / 'pizza.jpg'

if custom_img_path.exists():
    print('Image arleady exist...')

else:
    with open(custom_img_path, 'wb') as f:
        request = requests.get('https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcTOgO-n0NBof4CDo5jWSQtTJ_47meMAXKYgbVIEq89KybeL2h86HjjX2BEr9uHwe3qPTozPWl5r4-FZCZFMnkizkHPE15LUT8DkZh-Ld-j2jQ')
        f.write(request.content)
        print('Image has been downloaded')

custom_image = torchvision.io.read_image(custom_img_path).type(torch.float) / 255
custom_image.shape
custom_image.dtype
custom_image

plt.imshow(custom_image.permute(1, 2, 0))

transform = torchvision.transforms.Resize((64, 64))
custom_image_trans = transform(custom_image)

custom_image_pred = model_0(custom_image_trans.unsqueeze(0)).argmax(dim=1).item()


