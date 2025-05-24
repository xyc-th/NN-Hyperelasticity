import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import griddata
import torch
import torch.optim.lr_scheduler

# Set random seed
torch.manual_seed(2025)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
np.random.seed(2025)

# Physical settings
dim = 3
num_nodes_per_element = 4
voigt_map = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
num_element = 7762
num_node = 1661

# Training device, set to >=0 to use GPU-accelerated training
cuda_id = -1
if cuda_id >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{cuda_id}')
else:
    device = torch.device('cpu')
print(f"Setting training device to: {device}.")

# Path settings
data_dir = "../data"
output_dir = "../result"

# Dataset settings
material = 'Trivial-Test'
train_steps = ['1-1', '2-1', '3-1', '4-1', '5-1', '6-1', '7-1', '8-1', '9-1']
noise_type = 'disp_relative'
noise_level = 0.01
test1_steps = ['1-1', '2-1', '3-1', '4-1', '5-1', '6-1']
test2_steps = ['1-0', '1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10']

# Neural network training settings

'''
ensemble_size:                  Number of ICNNs in the ensemble
random_init: {True,False}       Randomly initialize weights and biases
n_input: (default = 3)          The three principal invariants
n_output: (default = 1)         The strain energy density 
n_hidden:                       List of number of neurons for each hidden layer
use_dropout: {True,False}       Use dropout in ICNN architecture
dropout_rate:                   Dropout probability
use_sftpSquared:                Use squared softplus activation for the hidden layers
scaling_sftpSq:                 Scale the output after squared softplus activation to mitigate exploding gradients
opt_method:                     Specify the NN optimizer
epochs:                         Number of epochs to train the ICNN
lr_schedule: {cyclic,multistep} Choose a learning rate scheduler to improve convergence and performance
eqb_loss_factor:                Factor to scale the force residuals at the free DoFs
bc_loss_factor:                 Factor to scale the force residuals at the fixed DoFs
verbose_frequency:              Prints the training progress every nth epoch
'''
ensemble_size = 10
epochs = 500
lr_schedule = 'plateau'
optimization_method = 'adam'
random_init_linear = True
n_input = 3
n_output = 1
n_hidden = [16, 64, 64, 16]
use_dropout = False
dropout_rate = 0.2
use_sftpSquared = True
scaling_sftpSq = 1.0 / 12
if lr_schedule == 'multistep':
    lr = 0.1
    le_milestones = [epochs // 4, epochs // 4 * 2, epochs // 4 * 3]
    lr_decay = 0.1
    cycle_momentum = False
elif lr_schedule == 'cyclic':
    base_lr = 0.001
    max_lr = 0.1
    lr = base_lr
    cycle_momentum = False
    step_size_up = 50
    step_size_down = 50
elif lr_schedule == 'cosine':
    lr = 0.001
    T_max = 50
    eta_min = 1e-6
elif lr_schedule == 'cosine_warm':
    lr = 0.01
    T_0 = epochs // 10
    T_mult = 2
    eta_min = 1e-6
elif lr_schedule == 'plateau':
    lr = 0.01
    factor = 0.9
    patience = 10
eqb_loss_factor = 1.0
bc_loss_factor = 1.0
verbose_frequency = 1

# Output settings
num_marker = 90

# Post process settings
conf_threshold = 2.0

# Plot settings
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
# plt.rcParams['text.usetex'] = True
