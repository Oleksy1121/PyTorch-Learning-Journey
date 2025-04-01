import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#%% DATA
w = 0.5
bias = 1.5

X = torch.arange(0.0, 2.0, 0.05)
y = X * w + bias

plt.scatter(X, y)
plt.show()

#%% split data
test_size = 0.2

X_train, y_train = X[: int((1-test_size) * len(X))], y[: int((1-test_size) * len(y))]
X_test,  y_test  = X[int((1-test_size) * len(X)) :], y[int((1-test_size) * len(y)) :]

#%% plot
def plot_data(y_pred = None):
    plt.scatter(X_train, y_train, c='b', alpha=0.8, label='Train data')
    plt.scatter(X_test, y_test, c='gray', alpha=0.8, label='Test data')
    
    if y_pred is not None:
        plt.scatter(X_test, y_pred, c='gold', label='Prediction')
    
    plt.legend()
    plt.show()
    
plot_data()

#%% linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

#%%
lr = LinearRegression()
lr.state_dict()

cost = nn.L1Loss()
optim = torch.optim.SGD(params=lr.parameters(), lr=0.01)

#%%
EPOCHS = 500

epoch_count = []
train_loss_values = []
test_loss_values = []


for epoch in range(EPOCHS):
    
    # train
    lr.train()
    y_pred = lr(X_train.unsqueeze(1))
    loss = cost(y_pred, y_train)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    
    # evaluation
    lr.eval()
    with torch.no_grad():
        test_pred = lr(X_test.unsqueeze(1))
        test_loss = cost(test_pred, y_test)
    
    if epoch % 10 == 1:
        epoch_count.append(epoch)
        train_loss_values.append(loss)
        test_loss_values.append(test_loss)
        
        print(f'Epoch: {epoch}, Train MAE: {loss:.3f}, Test MAE: {test_loss:.3f}')
    

plt.plot(epoch_count, torch.Tensor(train_loss_values), color='b', label='train')
plt.plot(epoch_count, torch.Tensor(test_loss_values), color='r', label='test')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()




















