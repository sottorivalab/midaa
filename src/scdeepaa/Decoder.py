import torch
import torch.nn as nn

from scdeepaa.Utils_net import build_last_layer_from_output_dist, build_sequential_layer, build_layer_general, calculate_flattened_size, calculate_flattened_size_decoder

class Decoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dims_dec_common,hidden_dims_dec_last, 
                 output_types_input,layers_independent_types = None,
                 input_size_aux = None, output_types_side = None, 
                 hidden_dims_dec_last_side = None, layers_independent_types_side = None, 
                 linearize = False,
                 image_size = [256,256], kernel_size=3, stride=1, padding=1, pool_size=2):
        
        super().__init__()
        ### Network attributes ###

        self.layers_common = nn.ModuleList()
        self.layers_independent_input = nn.ModuleList()
        
        if not isinstance(hidden_dims_dec_last[0], list):
            hidden_dims_dec_last = [hidden_dims_dec_last for i in range(len(input_size))]
        
        self.hidden_dims_dec_last = hidden_dims_dec_last
        self.hidden_dims_dec_common = hidden_dims_dec_common
        self.n_inputs = len(output_types_input)
        self.layers_independent_types = layers_independent_types
        
        self.n_layers_common = len(hidden_dims_dec_common)
        
        if self.layers_independent_types is None:
            self.layers_independent_types = ["linear"] * len(input_size)
        else:
            self.layers_independent_types = ["conv_transpose" if nn_type == "conv" else nn_type for nn_type in self.layers_independent_types]
        
        self.has_side = input_size_aux is not None
        
        self.n_layers_ind =  [len(ind_dim) for ind_dim in hidden_dims_dec_last]

        if self.has_side:
            self.n_side = len(output_types_side)
            self.layers_independent_side = nn.ModuleList()
            self.layers_independent_types_side = layers_independent_types_side
            
            if hidden_dims_dec_last_side is None:
                hidden_dims_dec_last_side = hidden_dims_dec_last
            
            if not isinstance(hidden_dims_dec_last_side[0], list):
                        hidden_dims_dec_last_side = [hidden_dims_dec_last_side for i in range(len(output_types_side))]
                    
                    
            
            if self.layers_independent_types_side is None:
                self.layers_independent_types_side = ["linear"] * self.n_side
            else:
                self.layers_independent_types_side = ["conv_transpose" if nn_type == "conv" else nn_type for nn_type in self.layers_independent_types_side]
                    
            self.hidden_dims_dec_last_side = hidden_dims_dec_last_side
            self.n_layers_ind_side =  [len(ind_dim) for ind_dim in hidden_dims_dec_last_side]
        
        self.output_types_input = output_types_input
        self.output_types_lateral = output_types_side
        self.input_size_aux = input_size_aux
        self.input_size = input_size
        self.linearize = linearize
        
        self.softmax = nn.Softmax(1)
        
        ### Layers building ###

        self.layers_common.append(build_sequential_layer(z_dim, hidden_dims_dec_common[0], linearize = self.linearize))


        if self.n_layers_common > 1:
            for i in range(self.n_layers_common - 1):
                self.layers_common.append(
                    build_sequential_layer(hidden_dims_dec_common[i], hidden_dims_dec_common[i + 1], linearize = self.linearize))
        
        for k in range(self.n_inputs):
            mod_list_tmp = nn.ModuleList()
            for lay in range(self.n_layers_ind[k] + 1):
                if  lay == self.n_layers_ind[k]:
                    if self.n_layers_ind[k] > 0:
                        mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_last[k][-1],input_size[k], output_types_input[k], self.layers_independent_types[k]))
                    else:
                        mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_common[-1],input_size[k], output_types_input[k], self.layers_independent_types[k]))
                elif lay == 0:
                    if self.layers_independent_types[k] == "conv_transpose" and self.n_layers_ind[k] > 0:
                        flat_size, flat_size_list = calculate_flattened_size_decoder(image_size, len(hidden_dims_dec_last[k]) - 1, hidden_dims_dec_last[k][0], kernel_size, stride, padding, pool_size)
                        self.flat_size = flat_size_list
                        mod_list_tmp.append(nn.Linear(hidden_dims_dec_common[-1], flat_size))
                    else:
                        mod_list_tmp.append(build_layer_general(hidden_dims_dec_common[-1],hidden_dims_dec_last[k][0], self.layers_independent_types[k], linearize = self.linearize))
                else:
                    mod_list_tmp.append(build_layer_general(hidden_dims_dec_last[k][lay-1],hidden_dims_dec_last[k][lay], self.layers_independent_types[k], linearize = self.linearize))
            self.layers_independent_input.append(mod_list_tmp)

        
        if self.has_side:
            for k in range(self.n_side):
                mod_list_tmp = nn.ModuleList()    
                for lay in range(self.n_layers_ind_side[k] + 1):
                    if  lay == self.n_layers_ind_side[k]:
                        if self.n_layers_ind_side[k] > 0:
                            mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_last_side[k][-1],input_size_aux[k], output_types_side[k], self.layers_independent_types_side[k]))
                        else:
                            mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_common[-1],input_size_aux[k], output_types_side[k],self.layers_independent_types_side[k]))
                    elif lay == 0:
                        if self.layers_independent_types_side[k] == "conv_transpose" and self.n_layers_ind_side[k] > 0:
                            # compute steps for final image size
                            flat_size, flat_size_list = calculate_flattened_size_decoder(image_size, len(hidden_dims_dec_last_side[k]) - 1, hidden_dims_dec_last_side[k][0], kernel_size, stride, padding, pool_size)
                            self.flat_size_side = flat_size_list
                            mod_list_tmp.append(nn.Linear(hidden_dims_dec_common[-1], flat_size))
                        else:
                            mod_list_tmp.append(build_layer_general(hidden_dims_dec_common[-1],hidden_dims_dec_last_side[k][0], 
                                                                self.layers_independent_types_side[k], linearize = self.linearize))
                    else:
                        mod_list_tmp.append(build_layer_general(hidden_dims_dec_last_side[k][lay-1],
                                                                hidden_dims_dec_last_side[k][lay], 
                                                                self.layers_independent_types_side[k], linearize = self.linearize))
                self.layers_independent_side.append(mod_list_tmp)

  

    def forward(self, z):
    # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.layers_common[0](z)
        
        
        
        if self.n_layers_common > 1:
            for lay in range(1, self.n_layers_common):
                hidden = self.layers_common[lay](hidden)
        
        
        
        input_reconstructed = [None] * self.n_inputs
        hidden_out_inp = [None] *  self.n_inputs
        for k in range(self.n_inputs):
            for lay in range(self.n_layers_ind[k] + 1):
                if lay == 0:
                    hidden_out_inp[k] = self.layers_independent_input[k][lay](hidden)
                elif lay == 1 and self.layers_independent_types[k] == "conv_transpose":
                    hidden_out_inp[k] = self.layers_independent_input[k][lay](hidden_out_inp[k].reshape([-1, self.flat_size[0], self.flat_size[1], self.flat_size[2]]))
                else:
                    hidden_out_inp[k] = self.layers_independent_input[k][lay](hidden_out_inp[k])
            input_reconstructed[k] = hidden_out_inp[k]

        if self.has_side:
            lateral_reconstructed = [None] * self.n_side
            hidden_out_side = [None] *  self.n_side
            for k in range(self.n_side):
                for lay in range(self.n_layers_ind_side[k] + 1):
                    if lay == 0:
                        hidden_out_side[k] = self.layers_independent_side[k][lay](hidden)
                    elif lay == 1 and self.layers_independent_types_side[k] == "conv_transpose":
                        hidden_out_side[k] = self.layers_independent_side[k][lay](hidden_out_side[k].reshape([-1, self.flat_size_side[0], self.flat_size_side[1], self.flat_size_side[2]]))
                    else:
                        hidden_out_side[k] = self.layers_independent_side[k][lay](hidden_out_side[k])
                lateral_reconstructed[k] = hidden_out_side[k]
            return input_reconstructed, lateral_reconstructed             
        return input_reconstructed, None