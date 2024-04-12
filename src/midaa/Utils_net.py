import torch
import torch.nn as nn
import math


def build_multimodal_linear_encoder():
    pass

def build_multimodal_linear_decoder():
    pass


def build_conv_layer(in_channels, out_channels, flatten = False, kernel_size=3, stride=1, padding=1, pool_size=2, pool_stride=2):
    if flatten:
        layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    #nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride),
                    nn.Flatten()
                )
    else:
        layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            )
    
    return layer

def build_conv_layer_reverse(in_channels, out_channels, flatten = False, kernel_size=3, stride=1, padding=1, pool_size=2, pool_stride=2):
    if flatten:
        layer = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    #nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=pool_size, mode='nearest'),
                    nn.Flatten()
                )
    else:
        layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Upsample(scale_factor=pool_size, mode='nearest')
            )
    
    return layer

def build_sequential_layer(dim1, dim2, linearize = False):
    if linearize:
        net = nn.Sequential(
            nn.Linear(dim1, dim2)
            #nn.ReLU()
        )
    else:
    
        net = nn.Sequential(
                nn.Linear(dim1, dim2),
                nn.Dropout(p=0.2),
                nn.BatchNorm1d(dim2),
                nn.Tanh()
                #nn.ReLU()
            )
    
    return net
        

def build_last_layer_from_output_dist(dim,input_size, output_type, nn_type, **kargs):
        if nn_type == "conv_transpose":
            return build_last_layer_from_output_dist_conv(dim,input_size, output_type, **kargs)
        elif nn_type == "linear":
            return build_last_layer_from_output_dist_flat(dim,input_size, output_type)

def build_last_layer_from_output_dist_flat(dim,input_size, output_type):       
        net = nn.Linear(dim, input_size)
        return net

def build_last_layer_from_output_dist_conv(dim,input_size, output_type, kernel_size=3, stride=1, padding=1):
        net = nn.Sequential(
            nn.ConvTranspose2d(dim, input_size, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Flatten(),
            nn.Sigmoid()
        )
        return net

    
def build_layer_general(in_channels, out_channels, nn_type,linearize = False, flatten = False, **kargs):
    if nn_type == "conv":
        return build_conv_layer(in_channels, out_channels, flatten = flatten, **kargs)
    elif nn_type == "linear":
        return build_sequential_layer(in_channels, out_channels, linearize = linearize)
    elif nn_type == "conv_transpose":
        return build_conv_layer_reverse(in_channels, out_channels, flatten = flatten, **kargs)
    else:
        print("Network type not support, choose one of [linear, conv]!")


def calculate_flattened_size(image_size, num_conv_layers, D,
                             filter_size, stride, padding, pool_size, pool_stride):
    W, H = image_size  # Width, Height, Depth of the input image

    for _ in range(num_conv_layers):
        W = math.floor((W - filter_size + 2*padding) / stride) + 1
        H = math.floor((H - filter_size + 2*padding) / stride) + 1
        W = math.floor((W - pool_size) / pool_stride) + 1
        H = math.floor((H - pool_size ) / pool_stride) + 1
        # Depth remains the same after pooling

    return W * H * D  # Flattened size


def calculate_flattened_size_decoder(image_size, num_conv_layers, D,
                             filter_size, stride, padding, pool_stride):
    W, H = image_size  # Width, Height, Depth of the input image

    for _ in range(num_conv_layers):
        W = math.floor((W - filter_size + 2*padding) / stride) + 1
        H = math.floor((H - filter_size + 2*padding) / stride) + 1
        W = math.floor(W  / pool_stride) 
        H = math.floor(H  / pool_stride) 
        # Depth remains the same after pooling

    return W * H * D, [D,W,H]