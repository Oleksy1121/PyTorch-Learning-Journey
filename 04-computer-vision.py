import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from time import time
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import seaborn as sns
import os

#%% Download the dataset
PATH = 'C:/Users/Marcin/Documents/Python/Udemy/PyTorch/data/'

train_data = datasets.FashionMNIST(root=PATH, train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = datasets.FashionMNIST(root=PATH, train=False, transform=torchvision.transforms.ToTensor(), download=True)

#%% check the data
image, label = train_data[0]
image.shape, label

train_data.targets
classes = train_data.classes
train_data.class_to_idx


#%% Visualize the data
plt.imshow(image.squeeze(), cmap='gray')
plt.axis(False)
plt.title(train_data.classes[label])

#%% visualise random probs
idx = torch.randint(0, len(train_data), (16, ))
plt.figure(figsize=(6, 7))
for i, _idx in enumerate(idx):
    image, label = train_data[_idx]
    plt.subplot(4, 4, i+1)
    plt.title(train_data.classes[label])
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis(False)
    plt.show()

#%% visualise random probs v2
cols, rows = 6,6
fig = plt.figure(figsize=(9, 9))

for i in range (rows*cols):
    random_idx = torch.randint(0, len(train_data)+1, (1,))
    image, label = train_data[i]

    fig.add_subplot(rows, cols, i+1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis(False)
    plt.title(train_data.classes[label])
    plt.show()

#%% dataloader
BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

len(train_dataloader), len(test_dataloader)


#%% check th dataloaders
train_feature_batch, train_labels_batch = next(iter(train_dataloader))

train_feature_batch
train_labels_batch

train_feature_batch.shape # give us shape of curent dataloader (e.g. 32, 1, 28, 28)
train_labels_batch.shape

image = train_feature_batch[0]
label = train_labels_batch[0]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(train_data.classes[label])
plt.axis(False)
plt.show()

#%% visualise dataloader
rows, cols = 2, 2
fig = plt.figure(figsize=(8, 8))

train_feature_batch, train_labels_batch = next(iter(train_dataloader))
for i in range(rows*cols):
    image, label = train_feature_batch[i], train_labels_batch[i]
    
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(image.squeeze())
    plt.title(train_data.classes[label])
    plt.axis(False)
    plt.show()

#%% check how its works Flatten layer
x = train_feature_batch[0] # first batch
x.shape # [1, 28, 28]

flatten_model = nn.Flatten() 

output = flatten_model(x)
output.shape # [1, 784]

#%% build baseline model
class FashionModel(nn.Module):
    def __init__(self, in_features :int, out_features :int, hidden_units :int):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_units),
            nn.Linear(hidden_units, out_features)
            )
    
    def forward(self, x):
        return self.layer_stack(x)

#%%
model_0 = FashionModel(in_features=784, out_features=len(train_data.classes), hidden_units=10)
model_0

model_0(output) # got logits ;)

#%% set loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters())

#%% training loop
EPOCHS = 10


evaluate_metrics = ['epoch_time_sec', 'total_time_sec', 
                    'train_loss' ,'train_acc', 'train_precision', 
                    'test_loss', 'test_acc',' test_precision']

metrics_df = pd.DataFrame(index=range(0, EPOCHS), columns=evaluate_metrics, dtype=float)


time_start = time()
for epoch in tqdm(range(EPOCHS)):
    
    # Train
    model_0.train()
    time_epoch = time()
    train_loss = 0
    train_acc = 0
    train_precision = 0
    
    for batch, (X_train, y_train) in enumerate(train_dataloader):
        
        # prediction
        y_pred = model_0(X_train) # it's logits
        
        # metrics
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        train_acc += accuracy_score(y_train.detach(), y_pred.argmax(dim=1))
        train_precision += precision_score(y_train.detach(), y_pred.argmax(dim=1), average='macro', zero_division=0)
        
        # gradient operations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # normalize metrics
    train_loss /= batch
    train_acc /= batch
    train_precision /= batch
    
    
    # Evaluation
    model_0.eval()
    test_loss = 0
    test_acc = 0
    test_precision=0
    
    for batch, (X_test, y_test) in enumerate(test_dataloader):
        
        # prediction
        with torch.inference_mode():
            test_pred = model_0(X_test)
        
        # metric
        test_loss += loss_fn(test_pred, y_test)
        test_acc += accuracy_score(y_test.detach(), test_pred.argmax(dim=1))
        test_precision += precision_score(y_test.detach(), test_pred.argmax(dim=1), average='macro', zero_division=0)
    
    # metric normalize
    test_loss /= batch
    test_acc /= batch
    test_precision /= batch
    
    # prints
    print(50*'-') if epoch != 0 else print()
    print(f'EPOCH: {epoch} | Computed in {(time() - time_epoch):.0f} seconds')
    print(f'TRAIN -> loss: {train_loss:.3f} | acc: {train_acc:.2f} | precision: {train_precision:.2f}')
    print(f'TEST  -> loss: {test_loss:.3f} | acc: {test_acc:.2f} | precision: {train_precision:.2f}')
        
    # update metric table
    metrics_df.loc[epoch] = [time()-time_epoch, time()-time_start,
                             train_loss.detach(), train_acc, train_precision,
                             test_loss, test_acc, test_precision]

    # final print
    if epoch == EPOCHS-1:
        time_end = time()
        print()
        print(50*'-')
        print(f'Total time of learning: {(time_end-time_start):.0f} seconds')
        
model_0_computing_time = time() - time_start
    
#%% evaluation function
def make_eval(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn):
    model.eval()
    
    # init metrics
    loss = 0
    acc = 0
    precision = 0
    
    # evaluation
    for X, y in tqdm(dataloader):
        with torch.inference_mode():
            y_pred= model(X)
        
        loss += loss_fn(y_pred, y)
        acc += accuracy_score(y, y_pred.argmax(dim=1))
        precision += precision_score(y, y_pred.argmax(dim=1), average='macro', zero_division=0)
    
    # normalize metrics
    loss /= len(dataloader)
    acc /= len(dataloader)
    precision /=len(dataloader)
    
        
    return {'model': model.__class__.__name__,
            'loss': loss.item(),
            'accuracy': acc,
            'precision': precision}
        
# check the evaluation of model
model_0_eval = make_eval(model_0, test_dataloader, loss_fn)

#%%

'''

NOW LET'S BUILD NON-LINEARITY MODEL'

'''

#%% set th device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
class FashionModelV2(nn.Module):
    def __init__(self, in_features :int, out_features :int, hidden_units :int):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features),
            nn.ReLU()
            )
    
    def forward(self, x):
        return self.layer_stack(x)
    
#%%
model_1 = FashionModelV2(in_features=28*28, out_features=len(train_data.classes), hidden_units=10).to(device)
model_1

#%% optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

#%% train function
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim,
               loss_fn,
               device=device):
    
    model.train()
    
    # init the train metrics
    train_loss, train_acc = 0, 0
    
    for batch, (X_train, y_train) in enumerate(data_loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        # prediction
        y_pred = model(X_train)
        
        # metrics
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        train_acc += accuracy_score(y_train.cpu(), y_pred.argmax(dim=1).cpu())
        
        # gradient operations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # normalize metrics
    train_loss /= batch
    train_acc /= batch
    
    return {'loss': train_loss.item(),
            'acc': train_acc}

#%% eval function
def eval_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn,
              device=device):
    
    model.eval()
    
    # init the metrics
    test_loss = 0
    test_acc = 0
    
    with torch.inference_mode():
        
        for batch, (X_test, y_test) in enumerate(data_loader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_score(y_test.cpu(), test_pred.argmax(dim=1).cpu())
        
        # normalize metrics
        test_loss /= batch
        test_acc /= batch
    
    return {'loss': test_loss.item(),
            'acc': test_acc}
            
#%% Training
EPOCHS = 10

time_start = time()
for epoch in tqdm(range(EPOCHS)):
    
    train_results = train_step(model=model_1,
                               data_loader=train_dataloader,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               device=device)
    
    eval_results = eval_step(model=model_1, 
                             data_loader=test_dataloader, 
                             loss_fn=loss_fn,
                             device=device)
    
    print()
    print(50*'-')
    print(f'Epoch: {epoch}')
    print('Train ->', train_results)
    print('Test  ->', eval_results)

model_1_computing_time = time() - time_start

model_1_eval = make_eval(model_1, test_dataloader, loss_fn)

#%%

'''

LET'S START WITH CON MODEL

'''

#%%
class FashionModelV3(nn.Module):
    def __init__(self, in_channels, hidden_units, out_channels):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=out_channels)
            )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f'conv_block_1 shape: {x.shape}')
        x = self.conv_block_2(x)
        # print(f'conv_block_2 shape: {x.shape}')
        x = self.classifier(x)
        # print(f'classifier shape: {x.shape}')
        return x
    
#%% init the model
model_2 = FashionModelV3(in_channels=1, hidden_units=10, out_channels=len(train_data.classes))
model_2    

#%%

'''
THIS BLOCK REPREZENT HOW IS CONVOLUTION LAYERS WORKS

'''

# read sample image
IMG_PATH = r'C:/Users/Marcin/Documents/Python/Udemy/PyTorch/data/sample.jpg'
img = cv2.imread(IMG_PATH)
img.shape

# transpose from 720x720x3 to 3x720x720
img = img.transpose(2, 0, 1)
img.shape

# to tensor
img = torch.tensor(img)
img = img.type(torch.float)
img.shape

# show the original image
plt.imshow(img.type(torch.long).numpy().transpose(1, 2, 0))
plt.axis(False)
plt.show()


'''Let's use convolution'''
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=5,
                       kernel_size=5,
                       stride=4,
                       padding=1)

img_conv = conv_layer(img)

# lets compare the data shapes
img.shape          # [3, 720, 720]
img_conv.shape     # [5, 180, 180]

img_conv[0].shape

# lets visualise
ROWS, COLS = 5, 1
fig = plt.figure(figsize=(3, 12))
for i in range(img_conv.shape[0]):
    fig.add_subplot(ROWS, COLS, i+1)
    plt.imshow(img_conv[i].detach())
    plt.title(i+1)
    plt.axis(False)


'''Let's use max pooling'''
max_pool_layer = nn.MaxPool2d(kernel_size=4)
img_max_pool = max_pool_layer(img_conv)


# lets compare the data shapes
img.shape          # [3, 720, 720]
img_conv.shape     # [5, 180, 180]
img_max_pool.shape # [5,  45,  45]


# lets visualise
fig = plt.figure(figsize=(3, 12))
for i in range(img_max_pool.shape[0]):
    fig.add_subplot(ROWS, COLS, i+1)
    plt.imshow(img_max_pool[i].detach())
    plt.title(i+1)
    plt.axis(False)



#%% lets pass the dummy data throught our model
feature_batch, labels_batch = next(iter(train_dataloader))

image = feature_batch[0]
image.shape

dummy_x = torch.randn(1, 1, 28, 28)
dummy_x.shape

model_2(dummy_x)

#%% set loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

#%% build a loop
EPOCHS = 10

time_start = time()
for epoch in tqdm(range(EPOCHS)):
    
    train_results = train_step(model_2, train_dataloader, optimizer, loss_fn, device)
    eval_results = eval_step(model_2, test_dataloader, loss_fn, device)
    
    print()
    print(50*'-')
    print(f'Epoch: {epoch}')
    print('Train ->', train_results)
    print('Test  ->', eval_results)

model_2_compute_time = time() - time_start
model_2_eval = make_eval(model_2, test_dataloader, loss_fn)
#%% compare models results
models_metric_df = pd.DataFrame([model_0_eval,
                                 model_1_eval,
                                 model_2_eval])

models_metric_df.set_index('model').plot(y='loss', title='Loss',  kind='barh')
models_metric_df.set_index('model').plot(y='accuracy', kind='barh')
models_metric_df.set_index('model').plot(y='precision', kind='barh')


models_metric_df['computing_time'] = [model_0_computing_time,
                                      model_1_computing_time,
                                      model_2_compute_time]

models_metric_df
models_metric_df.set_index('model').plot(y='computing_time',xlabel='seconds' , kind='barh')
models_metric_df.set_index('model').plot(y=['loss', 'accuracy', 'precision'], kind='barh')


model_2_eval



'''
MODELS RESULTS:


{'model': 'FashionModel',
 'loss': 0.6329792141914368,
 'accuracy': 0.7837460063897763,
 'precision': 0.7646190222627921}


{'model': 'FashionModelV2',
 'loss': 0.638441264629364,
 'accuracy': 0.7851437699680511,
 'precision': 0.720313259887275}


{'model': 'FashionModelV3',
 'loss': 0.2872351109981537,
 'accuracy': 0.8966653354632588,
 'precision': 0.8864130477655501}

'''


#%% prediction function
def make_prediction(model: torch.nn.Module,
                    list_of_samples):
    model.eval()
    pred_list = []
    
    with torch.inference_mode():
        for sample in tqdm(list_of_samples):
            y_logit = model(sample.unsqueeze(0))
            y_pred_prob = torch.softmax(y_logit, dim=1)
            y_pred = y_pred_prob.argmax(dim=1)
            pred_list.append(y_pred.item())
            
    return pred_list
            
#%% take the sample images
NO_OF_SAMPLES = 12

random_indexes = np.random.choice(len(test_data), (NO_OF_SAMPLES), replace=False)

random_images = [test_data[i][0] for i in random_indexes]
random_labels = [test_data[i][1] for i in random_indexes]

random_preds = make_prediction(model_2, random_images)

#%% plotting predictions
rows, cols = 4, 3

fig = plt.figure(figsize=[12, 9])
for i in range(NO_OF_SAMPLES):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(random_images[i].squeeze(), cmap='gray')
    plt.title(f'T: {test_data.classes[random_labels[i]]} | P: {test_data.classes[random_preds[i]]}')
    plt.axis(False)
plt.show()

#%% lets make prediction for our testing data
X = test_data.data.unsqueeze(1).type(torch.float)
y = test_data.targets.tolist()

X[0].shape
X.dtype

y_preds = make_prediction(model_2, X)

#%% lets compute and plot confussion matrix
cm = confusion_matrix(y, y_preds)
cm_normalized = confusion_matrix(y, y_preds, normalize='pred')

fig = plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.title('Confusion matrix')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.xticks(rotation=40)
plt.yticks(rotation=0)
plt.show()

#%% confussion matrix normalized
fig = plt.figure(figsize=(9, 9))
sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', 
            xticklabels=test_data.classes, yticklabels=test_data.classes)

plt.title('Confusion matrix - normalized')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.xticks(rotation=40)
plt.yticks(rotation=0)
plt.show()

#%% save model
model_2.state_dict()

MODEL_DIR = r'C:/Users/Marcin/Documents/Python/Udemy/PyTorch/models'
MODEL_NAME = 'Fashion_MNIST.pth'

PATH = os.path.normpath(os.path.join(MODEL_DIR, MODEL_NAME))

# save state dict
torch.save(model_2.state_dict(), PATH)

model_3 = FashionModelV3(in_channels=1, hidden_units=10, out_channels=len(train_data.classes))
model_3.load_state_dict(torch.load(PATH, weights_only=True))

torch.isclose(model_2.state_dict(), model_3.state_dict())


 

