import pyro
import pyro.distributions as dist
from torch.functional import F

import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO

import torch.distributions.constraints as constraints

import torch
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, input_dim, z_dim, hidden_dims_enc, bifurcate):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dims_enc[0])
        self.bc1 = nn.BatchNorm1d(hidden_dims_enc[0])
        
        self.fc2 = nn.Linear(hidden_dims_enc[0], hidden_dims_enc[1])
        self.bc2 = nn.BatchNorm1d( hidden_dims_enc[1])
        
        self.fc3 = nn.Linear(hidden_dims_enc[1], hidden_dims_enc[2])
        self.bc3 = nn.BatchNorm1d(hidden_dims_enc[2])
        
        self.fc11 = nn.Linear(hidden_dims_enc[2], z_dim + 1)
        self.fc12 = nn.Linear(hidden_dims_enc[2], z_dim + 1)
        self.fc13 = nn.Linear(hidden_dims_enc[2], z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.sofmax = nn.Softmax(dim=1)
        self.bifurcate = bifurcate
        self.softplus = nn.Softplus()
        
        if self.bifurcate:
          self.fc21 = nn.Linear(hidden_dims_enc[2], z_dim)
          self.fc22 = nn.Linear(hidden_dims_enc[2], z_dim)
          # self.fc23 = nn.Linear(hidden_dims_enc[2], z_dim)
          
    def forward(self, x):
        # define the forward computation on the sample x
        hidden1 = self.relu(self.bc1(self.fc1(x)))
        hidden2 = self.relu(self.bc2(self.fc2(hidden1)))
        hidden3 = self.relu(self.bc3(self.fc3(hidden2)))
        
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        A = self.sofmax(self.fc11(hidden3))
        B = self.sofmax(self.fc12(hidden3).t())
        # z_scale = self.softplus(self.fc13(hidden3))
        
        if self.bifurcate:
          A_2 = self.sofmax(self.fc21(hidden3))
          B_2 = self.sofmax(self.fc22(hidden3))
          #z_scale_2 = self.softplus(self.fc23(hidden3))
        
          return A, B, z_scale, A_2, B_2, z_scale_2
        return A, B
        # return A, B, z_scale