import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% introdution to tensors

scalar = torch.tensor(7)
scalar

scalar.ndim  #number of square brackets
scalar.shape #number of 


vector = torch.tensor([7, 14])
vector

vector.ndim
vector.shape


MATRIX = torch.tensor([[7, 8],
                       [9, 6]])
MATRIX

MATRIX.ndim
MATRIX.shape


MATRIX = torch.tensor([[1, 2],
                       [4, 5],
                       [6, 4],
                       [2, 2]])
MATRIX

MATRIX.ndim
MATRIX.shape

MATRIX[0]


TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [1, 2, 5]]])
TENSOR

TENSOR.ndim
TENSOR.shape #1 Big square, 4 Rows, 3 Cols


#%% random tensors

random_tensor = torch.rand(3, 4)
random_tensor

random_tensor.ndim
random_tensor.shape


random_tensor = torch.rand(224, 224, 3) # e.g. weights for images, height, width, RGB
random_tensor

random_tensor.ndim
random_tensor.shape


#%% zeros and ones

random_tensor = torch.rand(3, 4)

zeros = torch.zeros(3, 4)
zeros

zeros*random_tensor


ones = torch.ones(3, 4)
ones

ones*random_tensor

ones.dtype #chek data type, 

#%% creating a range of tensors

# torch.range()  - deprecated
# torch.arange() - works


one_to_ten = torch.arange(0, 10)
one_to_ten

torch.arange(start=0, end=100, step=10)

#%% creating tensors like

ten_zeros = torch.zeros_like(one_to_ten)
ten_zeros

#%% tensors dtypes

float_32_tensor = torch.tensor([1.0, 2.2, 3.5], 
                               dtype=None)
float_32_tensor
float_32_tensor.dtype


float_32_tensor = torch.tensor([1.0, 2.2, 3.5], 
                               dtype=torch.float16)
float_32_tensor
float_32_tensor.dtype


float_32_tensor = torch.tensor([1.0, 2.2, 3.5], 
                               dtype=None,            # tensor dtype
                               device=None,           # cpu / cuda / tpu
                               requires_grad=False)   # track gradient with this tensor operations
float_32_tensor
float_32_tensor.dtype



float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor

float_16_tensor * float_32_tensor


int_32_tensor = torch.tensor([1, 4, 2], dtype=torch.int32)
int_32_tensor

float_32_tensor * int_32_tensor



long_tensor = torch.tensor([1, 4, 2], dtype=torch.long)
long_tensor

float_32_tensor * long_tensor


#%% getting information from tensors - tensor atributes

float_32_tensor.dtype
float_32_tensor.device
float_32_tensor.shape


#%% manipulationg tensors


# addition
# subtraction
# multiplication
# division
# matrix multiplication

tensor = torch.tensor([1, 2, 3])
tensor

tensor + 10
tensor - 10
tensor * 5
tensor / 2

torch.add(tensor, 10)
tensor.add(10)

torch.subtract(tensor, 10)
tensor.subtract(10)

torch.multiply(tensor, 10)
tensor.multiply(10)


torch.divide(tensor, 2)
tensor.divide(2)


MATRIX_1 = torch.tensor([[2, 3, 4],
                         [3, 5, 2]])

MATRIX_2 = torch.tensor([[1, 2],
                         [4, 2],
                         [5, 2]])

torch.matmul(MATRIX_1, MATRIX_2) # 1D and more than 1D matrix multiplication
torch.dot(torch.tensor([1, 2, 3]), torch.tensor([5, 2, 3])) # only 1D matrix multiplication

torch.matmul(torch.tensor([1, 2, 3]), torch.tensor([5, 2, 3]))


# result of the matrix shapes (2, 3) @ (3, 2) is (2, 2)
# result of the matrix shapes (3, 2) @ (2, 3) is (3, 3)

torch.matmul(torch.rand(2, 3), torch.rand(3, 2))
torch.matmul(torch.rand(3, 2), torch.rand(2, 3))


#%% transpose

MATRIX_A = torch.tensor([[1, 2, 3],
                         [3, 4, 5]])


MATRIX_B = torch.tensor([[2, 3, 4],
                         [5, 3, 2]])


torch.matmul(MATRIX_A, MATRIX_B) # make shape ettot

MATRIX_B.T

torch.matmul(MATRIX_A, MATRIX_B.T)

#%% tensor agreggation

x = torch.arange(0, 100, 10)
x

x.min()
torch.min(x)
 
x.max()
torch.max(x)


# dtype error ocuured - FINALLY !!!
x.dtype
x.mean()
torch.mean(x) 

# need to convert to correct type
x.type(torch.float32).mean()
torch.mean(x.type(torch.float32))
torch.mean(x, dtype=torch.float32)


x.sum()
torch.sum(x)


#%% positional min and max indexes

x.argmin()
torch.argmin(x)

x.argmax()
torch.argmax(x)

#%% Reshaping, stacking, squezing and undsquezing

x = torch.arange(0, 10.)
x, x.shape

x_reshaped = x.reshape(2, 5)
x_reshaped, x_reshaped.shape

# view is only changing view of our x data but is the SAME MEMORY so if we change some value in view, the original vaue will also change
z = x.view(2, 5)
z, z.shape

z = x.view(1, 10)
z, z.shape
x
z


z[:, 5] = 50
x
z


z = x.view(2, 5)
z, z.shape
x
z

z[1, 0] = 20
x
z

# stack is just stacking data. dim argument is works like axis in numpy
x_stacked = torch.stack([torch.arange(0. , 10.),
                         torch.arange(10., 20.),
                         torch.arange(20., 30.),
                         torch.arange(30., 40.)])
x_stacked, x_stacked.shape

x_stacked = torch.stack([torch.arange(0. , 10.),
                         torch.arange(10., 20.),
                         torch.arange(20., 30.),
                         torch.arange(30., 40.)],
                        dim=0)
x_stacked, x_stacked.shape

x_stacked = torch.stack([torch.arange(0. , 10.),
                         torch.arange(10., 20.),
                         torch.arange(20., 30.),
                         torch.arange(30., 40.)],
                        dim=1)
x_stacked, x_stacked.shape


# squeze - deleting empty square brackets
x = torch.arange(0, 10)
x.shape

x.reshape(1, 10)
x.reshape(1, 10), x.reshape(1, 10).shape
x.reshape(1, 10).squeeze(), x.reshape(1, 10).squeeze().shape

x.reshape(2, 5)
x.reshape(2, 5), x.reshape(2, 5).shape
x.reshape(2, 5).squeeze(), x.reshape(2, 5).squeeze().shape


# unsqueze - is like making array drom vector (e.g. from shape [10] to shape [1, 10] or [10, 1])
x = torch.arange(0, 10)
x, x.shape

x.unsqueeze(dim=0), x.unsqueeze(dim=0).shape
x.unsqueeze(dim=1), x.unsqueeze(dim=1).shape


# torch permute - manimulate ordering of dimmension
x = torch.rand(2, 3, 5)
x, x.shape

x.permute((2, 0, 1)), x.permute((2, 0, 1)).shape

# example of the image
x_original = torch.rand(size=(8, 8, 3))  # (heigth, width, RGB)
x_original, x_original.shape

x_permuted = x_original.permute(2, 0, 1) # change order to (RGB, heigth, width)
x_permuted, x_permuted.shape

# need to remember that the permute is a view of origin data, so if we chage origin then permite also will change
x_original[0, 0, 0], x_permuted[0, 0, 0]

x_original[0, 0, 0] = 100
x_original[0, 0, 0], x_permuted[0, 0, 0]

#%% indexing

x = torch.arange(1, 10).reshape(1, 3, 3)
x, x.shape

x[0]         # getting first bracket of data
x[0, 1]      # getting middle row bracket
x[0, 1, 1]   # getting middle row, middle col

x[0, :, 1]   # getting middle col


#%% NumPy array to tensoor

array = np.arange(1., 10.)
tensor = torch.arange(1., 10.)

array, array.dtype    # NumPy origin type is 'float64'
tensor, tensor.dtype  # tensor origin type is tensor.float32

tensor_1 = torch.from_numpy(array)
tensor_1, tensor_1.dtype  # there need to be carefull because is converting float64 to tensor not float32


#%% reprodusibility (trying to make random out of random)

random_A = torch.rand(3, 4)
random_B = torch.rand(3, 4)

random_A
random_B
random_A == random_B


RANDOM_SEED = 42 # give us control for random numbers

torch.manual_seed(RANDOM_SEED) # its wokrs once only. so if we want to use next time we need to using again this func as below.
random_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_D = torch.rand(3, 4)

random_C
random_D
random_C == random_D


#%% using CUDA (colab version)

torch.cuda.is_available()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

torch.cuda.device_count()

# move tensor to cuda (if available)
tensor = torch.tensor([1, 2, 3]) # default CPU
teensor = tensor.to(device) # tensor device update

# if tensor is in gpu then we canc move back to NumPy
tensor.numpy() # make error if we had tensor GPU device
tensor.cpu().numpy()





