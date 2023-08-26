import torch
import torch.nn as nn


def build_sequential_layer(dim1, dim2):
    net = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU()
        )
    
    return net

def build_convolutional_layer(dim1, dim2):
    net = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU()
        )
    
    return net
        

def build_last_layer_from_output_dist(dim,input_size, output_type):
        net = None
        if output_type == "Normal" or output_type == "G" or \
        output_type == "Gaussian" or output_type == "N" or output_type == "NB" \
         or output_type == "Negative Binomial"  :
            net = nn.Linear(dim, input_size * 2)
        if output_type == "Categorical" or output_type == "Poisson" or\
            output_type == "C" or output_type == "P" or output_type == "Bernoulli" or output_type == "B":
            net = nn.Linear(dim, input_size)
        return net
