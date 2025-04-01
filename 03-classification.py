import torch
from torch import nn
from torcheval.metrics import R2Score
from sklearn.datasets import make_circles, make_blobs
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary
import pandas as pd
import seaborn as sns


#%% make and visualise data
X, y = make_circles(1000, noise=0.05)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, alpha=0.5)
plt.show()

#%% convert to tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.from_numpy(X).to(dtype=torch.float, device=device)
y = torch.from_numpy(y).to(dtype=torch.int, device=device)

X.dtype, y.dtype
X.device, y.device

#%% split to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)

#%% create class
class Circles(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.linear_01 = nn.Linear(in_features=2, out_features=5)
        self.linear_02 = nn.Linear(in_features=5, out_features=1)
        
    def forward(self, x):
        return self.linear_02(self.linear_01(x))


# this is second possibility of build simple model - use built-in function "nn.Sequential"
Circles_v2 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
    )


# this is combine method that we can us - custom class with "nn.Sequential"
class Circles_v3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.two_layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=5),
            nn.Linear(in_features=5, out_features=1)
            )
    
    def forward(self, x):
        return self.two_layers(x)

#%% make a instance of model
model_0 = Circles()
model_0.parameters
model_0.state_dict()

#%% try predictions (untrained model)
with torch.no_grad():
    y_pred = model_0(X_test)


y_pred[:5].squeeze()
y_test[:5]

#%% prediction steps
torch.eval()

# 1. Logits - its raw output from our model
with torch.no_grad():
    y_logits = model_0(X_test)
y_logits[:5].squeeze()


# 2. Prediction probabilities - Logits turned by activation function (e.g. Sigmoid)
y_pred_probs = torch.sigmoid(y_logits)
y_pred_probs[:5].squeeze()


# 3. Prediction - just round prediction probabilities
y_pred = torch.round(y_pred_probs)
y_pred[:5].squeeze()

#%% accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum()
    acc = correct / len(y_true)
    return acc


#%% setup loss function and optimizer
model_0 = Circles()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#%% build train / test loop
EPOCHS = 100

for epoch in range(EPOCHS):
    # Train
    model_0.train()
    
    # predictions
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # loss calculions
    loss = loss_fn(y_logits, y_train.float())
    acc = accuracy_fn(y_train, y_pred)
    
    # gradient operations
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    # Evaluation
    model_0.eval()
    with torch.no_grad():
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_pred, y_test.float())
        test_acc = accuracy_fn(y_test, test_pred)
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | loss: {loss:.3f} | test loss: {test_loss:.3f} | accuracy: {acc:.3f} | test accuracy: {test_acc:3f}')
    

#%%
'''
OUR model have loss 69 and test loss 77.
Its not enougth. 
Its propably because our model have only linear layers but problem is non-linear.

IMPROVMING MODEL STEPS:
1. Add one more linear layer 
2. Add more hidden units - from 5 to 10
3. Fit longer
4. Changing the activation functions
5. Changing the learning rate
6. Changing the loss function
'''

#%% Add one more linear layer & Add more hidden units - from 5 to 10
# create class
class Circles_v4(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_01 = nn.Linear(2, 10)
        self.linear_02 = nn.Linear(10, 10)
        self.linear_03 = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear_03(self.linear_02(self.linear_01(x)))

#%% create instance
model_1 = Circles_v4()
model_1.state_dict()
model_1.parameters

#%% set loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

# accuracy
def acc_fn(y_true, y_pred):
    return accuracy_score(y_true.detach().numpy(), y_pred.detach().numpy())



#%% train/test loop
EPOCHS = 1000

for epoch in range(EPOCHS):
    model_1.train()
    
    # prediction
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # loss
    loss = loss_fn(y_logits, y_train.float())
    acc = acc_fn(y_train, y_pred)
    
    # gradient calculation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    ## Evaluation
    model_1.eval()
    
    with torch.no_grad():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test.float())
        test_acc = acc_fn(y_test, test_pred)
        
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {acc:.2f} | Test accuracy: {test_acc:.2f}')
    
    
#%%

'''
OUR model have loss 69 and test loss 69.
Its not enougth. 
Its propably because our model have only linear layers but problem is non-linear.

NOW FOR TEST - create a linear model and try use our model on linear problem.
'''

#%% make linear data
w = 0.7
bias = 0.4 

X_lin = torch.arange(0, 1, 0.01)
y_lin = X_lin * w + bias

plt.scatter(X_lin, y_lin)
plt.show()

#%% split data
X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X_lin, y_lin, shuffle=False)

plt.scatter(X_lin_train, y_lin_train, label='Train data')
plt.scatter(X_lin_test, y_lin_test, label='Test data')
plt.show()

#%% R2 Score
def r2_fn(y_true, y_pred):
    return r2_score(y_true.squeeze().detach().numpy(), y_pred.squeeze().detach().numpy())

def r2_metric(y_true, y_pred):
    metric = R2Score()
    metric.update(y_pred, y_true)
    r2 = metric.compute()
    return r2


#%% create a model
Circle_v5 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
    ).to(device=device)


# create model instance
model_2 = Circle_v5
model_2.parameters

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.002)


#TRAIN
EPOCHS = 1000

X_lin_train, y_lin_train = X_lin_train.to(device), y_lin_train.to(device)
X_lin_test, y_lin_test = X_lin_test.to(device), y_lin_test.to(device)


for epoch in range(EPOCHS):
    model_2.train()
    
    # predictions
    y_pred = model_2(X_lin_train.unsqueeze(dim=1))
    
    # calculate loss
    #loss = loss_fn(y_pred.squeeze(), y_lin_train)
    loss = loss_fn(y_pred, y_lin_train.unsqueeze(dim=1))
    r2 = r2_fn(y_lin_train, y_pred)
    r2t = r2_metric(y_lin_train, y_pred.squeeze())
    
    # graadient operations
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    # Evaluation
    with torch.no_grad():
        test_pred = model_2(X_lin_test.unsqueeze(dim=1))
        #test_loss = loss_fn(test_pred.squeeze(), y_lin_test)
        test_loss = loss_fn(test_pred, y_lin_test.unsqueeze(dim=1))
        test_r2 = r2_fn(y_lin_test, test_pred)
        test_torch_r2 = r2_metric(y_lin_test, test_pred.squeeze())
    
    if epoch % 50 == 0:
        print(f'Epoch: {epoch} | MAE: {loss:.3f} | R2: {r2:.3f}| Test MAE: {test_loss:.3f} | Test R2: {test_r2:.3f}')
        print(f'Torch R2: {r2t:.3f} | Torch Test R2: {test_torch_r2:.3f}')

#%%
plt.scatter(X_lin_train, y_lin_train, c='b')
plt.scatter(X_lin_test, y_lin_test, c='gray')
plt.scatter(X_lin_test, model_2(X_lin_test.unsqueeze(dim=1)).detach().numpy(), c='gold')
plt.show()

#%%
'''
Linear problem solved well
got 0.04 MAE

So now we sure that problem exist in our model.
Lets make NON-LINEAR model
'''

#%% make circles data
X, y = make_circles(1000, noise=0.05)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X.dtype, y.dtype

#%% visualise
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, alpha=0.5)
plt.scatter

#%% split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

#%% create a non-linear model

class Circle_v6(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.linear_1 = nn.Linear(in_features=2, out_features=10)
        self.linear_2 = nn.Linear(in_features=10, out_features=10)
        self.linear_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(x)))))

#%% Accuracy function

def accuracy_metric(y_true, y_pred):
    return accuracy_score(y_true.detach(), y_pred.detach())

#%% init the model
model_3 = Circle_v6().to(device)
model_3.parameters

# set loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.5)

# train/eval loop
EPOCHS = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(EPOCHS):
    
    # Train
    model_3.train()
    
    # predictions
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_metric(y_train, y_pred)
    
    # gradient operations
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    # Eval
    model_3.eval()
    
    with torch.no_grad():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_metric(y_test, test_pred)
        
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.3f}  | Accuracy: {acc:.2f} | Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}")

#%% visualise
model_3.eval()
with torch.no_grad():
    y_pred = torch.round(torch.sigmoid(model_3(X_test).squeeze()))

y_pred

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=plt.cm.Set1, alpha=0.1, s=100)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=plt.cm.Set1, alpha=0.4, s=60)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap=plt.cm.Set1, alpha=1, s=20)
plt.show()


#%% visualise, visualise
plot_decision_boundary(model=model_3, X=X_train, y=y_train)

#%% 
'''
Lets check the activation functions how they works
'''

# simple data
A = torch.arange(-10, 10, 1)
plt.plot(A)

# Relu
plt.plot(torch.relu(A)) # ReLU just set values at minus as 0

plt.plot(torch.maximum(torch.Tensor([0]), A)) # its how ReLu works

# Sigmoid
plt.plot(torch.sigmoid(A))
plt.plot(1/(1 + torch.exp(-A)))


#%%
'''

LET'S START WITH MULTICLASS CLASSIFICATION !!!

'''
#%% make data
X_blob, y_blob = make_blobs(n_samples=1000, n_features=2, cluster_std=2, centers=4, random_state=42)

plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.Set1)
plt.show()

# transfer to tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

X_blob.dtype, y_blob.dtype

# train test split
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob)

#%% make blobs class
X_blob.shape[1] # check in_fetures
len(y_blob.unique()) # check out_fetures

class Blobs(nn.Module):
    def __init__(self, in_features=2, out_features=4, hidden_units=8):
        super().__init__()
        
        self.linear_layer_pack = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.Linear(hidden_units, hidden_units),
            nn.Linear(hidden_units, out_features)
            )
    
    def forward(self, x):
        return self.linear_layer_pack(x)

#%% iniciate model and set loss & optimizer
model_4 = Blobs()
next(model_4.parameters())
model_4.state_dict()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters())

#%% lets check predictions flow
'probs -> prediction pobabilities -> prediction labels'
model_4.eval()

with torch.inference_mode():
    y_logits = model_4(X_blob_test)

y_logits[:5], y_blob_test[:5] 


# pred probs
y_pred_probs = torch.softmax(y_logits, dim=1)
y_pred_probs[:5]


# prediction labels
y_pred = torch.argmax(y_pred_probs, dim=1)
y_pred[:5]


#%% lets write blobs train and evaluation loop
model_4 = Blobs()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.01)


EPOCHS = 1000

for epoch in range(EPOCHS):
    # Train
    model_4.train()
    
    # predictions
    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    # calculate the loss
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_metric(y_blob_train, y_pred)
    
    # gradient operations
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    
    
    # Evaluation
    model_4.eval()
    
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_metric(y_blob_test, test_pred)
    
    if epoch % 50 == 0:
        print(f'Epoch: {epoch} | loss: {loss:.4f} | acc: {acc:.2f} | test loss: {test_loss:.4f} | test acc: {test_acc:.2f}')

#%% visulize, visulize, visulize
plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
plt.title('Train data')
plot_decision_boundary(model_4, X_blob_train, y_blob_train)

plt.subplot(1,2,2)
plt.title('Test data')
plot_decision_boundary(model_4, X_blob_test, y_blob_test)
plt.show()

#%%

'''

Let's make nnon linear multiclassifiction model'

'''

#%% metrics
def precision_metric(y_true, y_pred, average):
    return precision_score(y_true.detach(), y_pred.detach(), average=average, zero_division=0)

def recall_metric(y_true, y_pred, average):
    return recall_score(y_true.detach(), y_pred.detach(), average=average, zero_division=0)

def f1_metric(y_true, y_pred, average):
    return f1_score(y_true.detach(), y_pred.detach(), average=average, zero_division=0)

def cm_metric(y_true, y_pred, normalize=None):
    return confusion_matrix(y_true.detach(), y_pred.detach(), normalize=normalize)


#%% build non linear model
blobs_v2 = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    )

#%% init the model
model_5 = blobs_v2
model_5

# loss & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_5.parameters(), lr=0.1)

# train and test loop
EPOCHS = 100

metric_applied = ['train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1'
                  'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']

metric_df = pd.DataFrame(index=range(EPOCHS), columns=metric_applied, dtype='float32')

for epoch in range(EPOCHS):
    
    # Train
    model_5.train()
    
    # prediction
    y_logits = model_5(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    
    # metrics
    loss = loss_fn(y_logits, y_blob_train)
    
    acc = accuracy_metric(y_blob_train, y_pred)
    precision = precision_metric(y_blob_train, y_pred, 'macro')
    recall = recall_metric(y_blob_train, y_pred, 'macro')
    f1 = f1_metric(y_blob_train, y_pred, 'macro')
    
    # gradient operations
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    # Evaluation
    model_5.eval()
    
    with torch.inference_mode():
        test_logits = model_5(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_metric(y_blob_test, test_pred)
        test_precision = precision_metric(y_blob_test, test_pred, 'macro')
        test_recall = recall_metric(y_blob_test, test_pred, 'macro')
        test_f1 = f1_metric(y_blob_test, test_pred, 'macro')
    
    
        # save logs to DataFrame
        metric_df.loc[epoch] = [loss.detach().numpy(), acc, precision, recall, f1, 
                                test_loss.detach().numpy(), test_acc, test_precision,
                                test_recall, test_f1]
    
    # print logs
    if epoch % 10 == 0:
        
        print(80*'-')
        print(f'EPOCH: {epoch}')
        print(f'TRAIN -> loss: {loss:.4f} | acc: {acc:.2f} | precision: {precision:.2f} | recall: {recall:.2f} | f1: {f1:.2f}')
        print(f'TEST  -> loss: {test_loss:.4f} | acc: {test_acc:.2f} | precision: {test_precision:.2f} | recall: {test_recall:.2f} | f1: {test_f1:.2f}')
        print(80*'-')
        print()
        
    
#%% visualize
plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plot_decision_boundary(model_5, X_blob_train, y_blob_train)
plt.title('Train data')
plt.subplot(1, 2, 2)
plot_decision_boundary(model_5, X_blob_test, y_blob_test)
plt.title('Test data')
plt.show()

#%% plot metrics

plt.plot(metric_df.index, metric_df['train_loss'], c='gray', label='Train')
plt.plot(metric_df.index, metric_df['test_loss'], c='b', label='Test')
plt.title('Loss Metric')
plt.legend()
plt.show()


plt.plot(metric_df.index, metric_df['train_accuracy'], c='gray', label='Train')
plt.plot(metric_df.index, metric_df['test_accuracy'], c='b', label='Test')
plt.title('Accuracy')
plt.legend()
plt.show()


plt.plot(metric_df.index, metric_df['train_precision'], c='gray', label='Train')
plt.plot(metric_df.index, metric_df['test_precision'], c='b', label='Test')
plt.title('Precision')
plt.legend()
plt.show()


plt.plot(metric_df.index, metric_df['train_recall'], c='gray', label='Train')
plt.plot(metric_df.index, metric_df['test_recall'], c='b', label='Test')
plt.title('Recall')
plt.legend()
plt.show()


plt.plot(metric_df.index, metric_df['train_f1'], c='gray', label='Train')
plt.plot(metric_df.index, metric_df['test_f1'], c='b', label='Test')
plt.title('F1 Score')
plt.legend()
plt.show()


#%% confussion matrix
model_5.eval()
with torch.inference_mode():
    y_preds_train = torch.softmax(model_5(X_blob_train), dim=1).argmax(dim=1)
    y_preds_test = torch.softmax(model_5(X_blob_test), dim=1).argmax(dim=1)

# calculate confusion matrix
cm_train = cm_metric(y_blob_train, y_preds_train)
cm_test = cm_metric(y_blob_test, y_preds_test)

# visualise train matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='.0f', cmap='Blues')
plt.title('Confussion matrix - Train')
plt.ylabel('True Label')
plt.xlabel('Prediction Label')

# visualise test matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='.0f', cmap='Blues')
plt.title('Confussion matrix - Test')
plt.ylabel('True Label')
plt.xlabel('Prediction Label')


#%% confusion matrix normalized
cm_train_norm = cm_metric(y_blob_train, y_preds_train, 'true')
cm_test_norm = cm_metric(y_blob_test, y_preds_test, 'true')

# visualise train matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_train_norm, annot=True, fmt='.2f', cmap='Blues')
plt.title('Confussion matrix normalized - Train')
plt.ylabel('True Label')
plt.xlabel('Prediction Label')

# visualise test matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test_norm, annot=True, fmt='.2f', cmap='Blues')
plt.title('Confussion matrix normalized - Test')
plt.ylabel('True Label')
plt.xlabel('Prediction Label')


