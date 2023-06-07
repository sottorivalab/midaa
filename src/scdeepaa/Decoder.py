import torch
import torch.nn as nn

from scdeepaa.Utils_net import build_last_layer_from_output_dist, build_sequential_layer

class Decoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dims_dec_common,hidden_dims_dec_last, 
                 output_types_input,input_size_aux = None, output_types_side = None):
        super().__init__()
        ### Network attributes ###

        self.layers_common = nn.ModuleList()
        self.layers_independent_input = nn.ModuleList()
        self.hidden_dims_dec_last = hidden_dims_dec_last
        self.hidden_dims_dec_common = hidden_dims_dec_common
        self.n_inputs = len(output_types_input)
        
        self.n_layers_common = len(hidden_dims_dec_common)

        
        self.has_side = input_size_aux is not None
        self.n_layers_ind = len(hidden_dims_dec_last)

        if self.has_side:
            self.n_side = len(output_types_side)
            self.layers_independent_side = nn.ModuleList()
        
        self.output_types_input = output_types_input
        self.output_types_lateral = output_types_side
        self.input_size_aux = input_size_aux
        self.input_size = input_size
        
        self.softmax = nn.Softmax(1)
        
        ### Layers building ###

        self.layers_common.append(build_sequential_layer(z_dim, hidden_dims_dec_common[0]))


        if self.n_layers_common > 1:
            for i in range(self.n_layers_common - 1):
                self.layers_common.append(
                    build_sequential_layer(hidden_dims_dec_common[i], hidden_dims_dec_common[i + 1]))
        
        for k in range(self.n_inputs):
            mod_list_tmp = nn.ModuleList()
            for lay in range(self.n_layers_ind + 1):
                if  lay == self.n_layers_ind:
                    if self.n_layers_ind > 0:
                        mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_last[-1],input_size[k], output_types_input[k]))
                    else:
                        mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_common[-1],input_size[k], output_types_input[k]))
                elif lay == 0:
                    mod_list_tmp.append(build_sequential_layer(hidden_dims_dec_common[-1],hidden_dims_dec_last[0]))
                else:
                    mod_list_tmp.append(build_sequential_layer(hidden_dims_dec_last[lay-1],hidden_dims_dec_last[lay]))
            self.layers_independent_input.append(mod_list_tmp)

        
        if self.has_side:
            for k in range(self.n_side):
                mod_list_tmp = nn.ModuleList()
                for lay in range(self.n_layers_ind + 1):
                    if  lay == self.n_layers_ind:
                        if self.n_layers_ind > 0:
                            mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_last[-1],input_size_aux[k], output_types_side[k]))
                        else:
                            mod_list_tmp.append(build_last_layer_from_output_dist(hidden_dims_dec_common[-1],input_size_aux[k], output_types_side[k]))
                    elif lay == 0:
                        mod_list_tmp.append(build_sequential_layer(hidden_dims_dec_common[-1],hidden_dims_dec_last[0]))
                    else:
                        mod_list_tmp.append(build_sequential_layer(hidden_dims_dec_last[lay-1],hidden_dims_dec_last[lay]))
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
            for lay in range(self.n_layers_ind + 1):
                if lay == 0:
                    hidden_out_inp[k] = self.layers_independent_input[k][lay](hidden)
                else:
                    hidden_out_inp[k] = self.layers_independent_input[k][lay](hidden_out_inp[k])
            input_reconstructed[k] = hidden_out_inp[k]

        if self.has_side:
            lateral_reconstructed = [None] * self.n_side
            hidden_out_side = [None] *  self.n_side
            for k in range(self.n_side):
                for lay in range(self.n_layers_ind + 1):
                    if lay == 0:
                        hidden_out_side[k] = self.layers_independent_side[k][lay](hidden)
                    else:
                        hidden_out_side[k] = self.layers_independent_side[k][lay](hidden_out_side[k])
                lateral_reconstructed[k] = hidden_out_side[k]
            return input_reconstructed, lateral_reconstructed             
        return input_reconstructed, None