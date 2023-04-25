import pyro
import pyro.distributions as dist
from torch.functional import F

import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO

import torch.distributions.constraints as constraints

import torch
import torch.nn as nn

class Decoder(nn.Module):
  def __init__(self, input_size, z_dim, hidden_dims_dec, bifurcate, output_type,input_size_aux = None, trifurcate = False):
      super().__init__()
      # setup the linear transformations used
      self.fc1 = nn.Linear(z_dim, hidden_dims_dec[0])
      self.fc2 = nn.Linear(hidden_dims_dec[0], hidden_dims_dec[1])
      self.fc3 = nn.Linear(hidden_dims_dec[1], hidden_dims_dec[2])
      self.fc11 = nn.Linear(hidden_dims_dec[2], input_size)
      
      # setup the non-linearities
      self.relu = nn.ReLU()
      self.sofmax = nn.Softmax()
      self.bifurcate = bifurcate
      self.output_type = output_type
      self.trifurcate = trifurcate
      self.input_size_aux = input_size_aux
      
      if bifurcate:
        self.fc21 = nn.Linear(hidden_dims_dec[2], input_size_aux[0])
        if self.output_type == "gaussian":
          self.fc22 = nn.Linear(hidden_dims_dec[2], input_size_aux[0])
      if trifurcate:
        self.fc21 = nn.Linear(hidden_dims_dec[2], input_size_aux[0])
        self.fc31 = nn.Linear(hidden_dims_dec[2], input_size_aux[1])
        # if self.output_type[0] == "gaussian":
        #   self.fc22 = nn.Linear(hidden_dims_dec[2], input_size_aux[0])
        # if self.output_type[1] == "gaussian":
        #   self.fc32 = nn.Linear(hidden_dims_dec[2], input_size_aux[1])
          
  def forward(self, z):
      # define the forward computation on the latent z
      # first compute the hidden units
      hidden1 = self.relu(self.fc1(z))
      hidden2 = self.relu(self.fc2(hidden1))
      hidden3 = self.relu(self.fc3(hidden2))
      # return the parameter for the output
      loc_x = self.fc11(hidden3)
      
      if self.bifurcate:
        if self.output_type == "categorical":
          loc_y = self.sofmax(self.fc21(hidden3))
          return loc_x, loc_y
        if self.output_type == "gaussian":
          loc_y = self.fc21(hidden3)
          # sigma_y = torch.exp(self.fc22(hidden3))
          # return loc_x, loc_y, sigma_y
          return loc_x, loc_y
        
      if self.trifurcate:
        if self.output_type[0] == "categorical" and self.output_type[1] == "categorical":
          loc_y = self.sofmax(self.fc21(hidden3))
          loc_v =  self.sofmax(self.fc31(hidden3))
          return loc_x, loc_y, loc_v
        if self.output_type[0] == "gaussian" and self.output_type[1] == "gaussian":
          loc_y = self.fc21(hidden3)
          loc_v =  self.fc31(hidden3)
          # sigma_y = torch.exp(self.fc22(hidden3))
          # sigma_v = torch.exp(self.fc32(hidden3))
          # return loc_x, loc_y, loc_v, sigma_y, sigma_v
          return loc_x, loc_y, loc_v
        if self.output_type[0] == "gaussian" and self.output_type[1] == "categorical":
          loc_y = self.fc21(hidden3)
          loc_v =  self.sofmax(self.fc31(hidden3))
          # sigma_y = torch.exp(self.fc22(hidden3))
          # return loc_x, loc_y, loc_v, sigma_y
          return loc_x, loc_y, loc_v
        if self.output_type[0] == "categorical" and self.output_type[1] == "gaussian":
          loc_y = self.sofmax(self.fc21(hidden3))
          loc_v =  self.fc31(hidden3)
          # sigma_v = torch.exp(self.fc32(hidden3))
          # return loc_x, loc_y,loc_v, sigma_v
          return loc_x, loc_y, loc_v
        
      return loc_x
    