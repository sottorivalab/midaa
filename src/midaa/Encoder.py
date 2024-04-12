import torch
import torch.nn as nn

from midaa.Utils_net import build_sequential_layer, build_layer_general, calculate_flattened_size
from midaa.Utils import create_z_fix


class Encoder(nn.Module):
    
    def __init__(self, input_size, z_dim, hidden_dims_enc_common,
                 hidden_dims_enc_ind, hidden_dims_enc_pre_Z,
                 fix_Z, linearize = False,
                 layers_independent_types = None, 
                 image_size = [256,256], kernel_size=3, stride=1, 
                 padding=1, pool_size=2, pool_stride=2):
        super().__init__()
        # setup the three linear transformations used
        self.layers_common = nn.ModuleList()
        self.layers_independent_input = nn.ModuleList()
        self.layers_latent = nn.ModuleList()
        if layers_independent_types is None:
            layers_independent_types = ["linear"] * len(input_size)
            
        if not isinstance(hidden_dims_enc_ind[0], list):
            hidden_dims_enc_ind = [hidden_dims_enc_ind for i in range(len(input_size))]
            
        self.layers_independent_types = layers_independent_types
        self.n_inputs = len(input_size)
        self.n_layers_common = len(hidden_dims_enc_common)
        self.n_layers_ind = [len(ind_dim) for ind_dim in hidden_dims_enc_ind]
        self.n_layers_pre_Z = len(hidden_dims_enc_pre_Z)
        self.fix_Z = fix_Z
        self.linearize = linearize
        
        if self.fix_Z:
            self.fixed_Z = create_z_fix(z_dim)

        for k in range(self.n_inputs):
            mod_list_tmp = nn.ModuleList()
            for lay in range(self.n_layers_ind[k]):
                if lay == 0:
                    if self.n_layers_ind[k] > 0:
                        if self.n_layers_ind[k] == 1:
                            mod_list_tmp.append(build_layer_general(input_size[k], hidden_dims_enc_ind[k][0], self.layers_independent_types[k],  
                                                                    linearize = linearize,
                                                                    flatten = True, kernel_size=3, stride=1, padding=1, pool_size=2, pool_stride=2))
                        else:
                            mod_list_tmp.append(build_layer_general(input_size[k], hidden_dims_enc_ind[k][0], self.layers_independent_types[k], 
                                                                    linearize = linearize,
                                                                    kernel_size=3, stride=1, padding=1, pool_size=2, pool_stride=2))
                    else:
                        mod_list_tmp.append(build_layer_general(input_size[k], hidden_dims_enc_common[0], self.layers_independent_types[k], 
                                                                linearize = linearize,
                                                                flatten = True, kernel_size=3, stride=1, padding=1, pool_size=2, pool_stride=2))
                elif lay == self.n_layers_ind[k] - 1:
                    mod_list_tmp.append(build_layer_general(hidden_dims_enc_ind[k][lay-1],hidden_dims_enc_ind[k][lay], 
                                                            self.layers_independent_types[k], linearize = linearize,
                                                            flatten = True, kernel_size=3, 
                                                            stride=1, padding=1, pool_size=2, pool_stride=2))
                else:
                    mod_list_tmp.append(build_layer_general(hidden_dims_enc_ind[k][lay-1],hidden_dims_enc_ind[k][lay],
                                                            self.layers_independent_types[k], linearize = linearize,
                                                            kernel_size=3, 
                                                            stride=1, padding=1, pool_size=2, pool_stride=2))
            self.layers_independent_input.append(mod_list_tmp)
        sum_flat = 0
        sum_conv = 0
        if "linear" in layers_independent_types:
            sum_flat = sum([hidden_dims_enc_ind[i][-1] for i in range(len(hidden_dims_enc_ind)) if layers_independent_types[i] == "linear"])
        if "conv" in layers_independent_types:
            sum_conv = sum([calculate_flattened_size(image_size, len(hidden_dims_enc_ind[i]), hidden_dims_enc_ind[i][-1], kernel_size, stride, padding, pool_size, pool_stride) for i in range(len(hidden_dims_enc_ind)) if layers_independent_types[i] == "conv"])
        for i in range(self.n_layers_common):
            if i == 0:
                self.layers_common.append(
                    build_sequential_layer(sum_flat + sum_conv, hidden_dims_enc_common[i], linearize = linearize)
                    )
            else:
                self.layers_common.append(
                    build_sequential_layer(hidden_dims_enc_common[i-1], hidden_dims_enc_common[i], linearize = linearize)
                    )
        dims_z = [z_dim + 1, z_dim + 1, z_dim]

        for k in range( 3 - int(self.fix_Z) ):
            mod_list_tmp = nn.ModuleList()
            for lay in range(self.n_layers_pre_Z + 1):
                if lay == 0:
                    if self.n_layers_pre_Z > 0:
                        mod_list_tmp.append(build_sequential_layer(hidden_dims_enc_common[-1], hidden_dims_enc_pre_Z[0], linearize = linearize))
                    else:
                        mod_list_tmp.append(nn.Linear(hidden_dims_enc_common[-1], dims_z[k]))
                elif lay == self.n_layers_pre_Z:
                    mod_list_tmp.append(nn.Linear(hidden_dims_enc_pre_Z[-1],dims_z[k]))
                else:
                    mod_list_tmp.append(build_sequential_layer(hidden_dims_enc_pre_Z[lay-1],hidden_dims_enc_pre_Z[lay], linearize = linearize))
            self.layers_latent.append(mod_list_tmp)

        
        self.sofmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        #self.sigmoid = lambda x:  x / torch.max(x, axis = 1)
          
    def forward(self, x):
        # define the forward computation on the sample x
        
        independent_flow = [None] * self.n_inputs
        independent_flow_tmp = [None] *  self.n_inputs
        for k in range(self.n_inputs):
            for lay in range(self.n_layers_ind[k]):
                if lay == 0:
                    independent_flow_tmp[k] = self.layers_independent_input[k][lay](x[k])
                else:
                    independent_flow_tmp[k] = self.layers_independent_input[k][lay](independent_flow_tmp[k])
            independent_flow[k] = independent_flow_tmp[k]
        
        
        to_common = torch.hstack(independent_flow)

        for lay in range(self.n_layers_common):
            to_common = self.layers_common[lay](to_common)
        
        if not self.fix_Z:
            latent_variables = [None] * 3
            latent_variables_tmp = [None] *  3
            for k in range(3):
                for lay in range(self.n_layers_pre_Z +1):
                    if lay == 0:
                        latent_variables_tmp[k] = self.layers_latent[k][lay](to_common)
                    else:
                        latent_variables_tmp[k] = self.layers_latent[k][lay](latent_variables_tmp[k])
                latent_variables[k] = latent_variables_tmp[k]



            A = self.sofmax(latent_variables[0])
            B = self.sofmax(latent_variables[1].t())
            Z = latent_variables[2]
            if not self.linearize:
                #Z = Z / Z.abs().max(dim=0).values
                Z = Z / (Z.abs().max(dim=0).values + 1e-9)
                
            
        else:
            latent_variables = [None] * 2
            latent_variables_tmp = [None] *  2
            for k in range(2):
                for lay in range(self.n_layers_pre_Z +1):
                    if lay == 0:
                        latent_variables_tmp[k] = self.layers_latent[k][lay](to_common)
                    else:
                        latent_variables_tmp[k] = self.layers_latent[k][lay](latent_variables_tmp[k])
                latent_variables[k] = latent_variables_tmp[k]
            A = self.sofmax(latent_variables[0])
            B = self.sofmax(latent_variables[1].t())
            Z = self.fixed_Z
       
        return A, B, Z 
