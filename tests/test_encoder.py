import torch
import torch.nn as nn

def build_sequential_layer(dim1, dim2):
    net = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.LeakyReLU()
        )
    
    return net

class Encoder(nn.Module):
    
    def __init__(self, input_size, z_dim, hidden_dims_enc_common,hidden_dims_enc_ind, hidden_dims_enc_pre_Z):
        super().__init__()
        # setup the three linear transformations used
        self.layers_common = nn.ModuleList()
        self.layers_independent_input = nn.ModuleList()
        self.layers_latent = nn.ModuleList()
        self.n_inputs = len(input_size)
        self.n_layers_common = len(hidden_dims_enc_common)
        self.n_layers_ind = len(hidden_dims_enc_ind)
        self.n_layers_pre_Z = len(hidden_dims_enc_pre_Z)

        for k in range(self.n_inputs):
            mod_list_tmp = nn.ModuleList()
            for lay in range(self.n_layers_ind):
                if lay == 0:
                    if self.n_layers_ind > 0:
                        mod_list_tmp.append(build_sequential_layer(input_size[k], hidden_dims_enc_ind[0]))
                    else:
                        mod_list_tmp.append(build_sequential_layer(input_size[k], hidden_dims_enc_common[0]))
                elif lay == self.n_layers_ind + 1:
                    mod_list_tmp.append(build_sequential_layer(hidden_dims_enc_ind[-1],hidden_dims_enc_common[0]))
                else:
                    mod_list_tmp.append(build_sequential_layer(hidden_dims_enc_ind[lay-1],hidden_dims_enc_ind[lay]))
            self.layers_independent_input.append(mod_list_tmp)

        for i in range(self.n_layers_common):
            if i == 0:
                self.layers_common.append(
                    build_sequential_layer(hidden_dims_enc_ind[-1] * self.n_inputs, hidden_dims_enc_common[i])
                    )
            else:
                self.layers_common.append(
                    build_sequential_layer(hidden_dims_enc_common[i-1], hidden_dims_enc_common[i])
                    )
        
        dims_z = [z_dim + 1, z_dim + 1, z_dim]

        for k in range(3):
            mod_list_tmp = nn.ModuleList()
            for lay in range(self.n_layers_pre_Z + 1):
                print(lay)
                if lay == 0:
                    if self.n_layers_pre_Z > 0:
                        mod_list_tmp.append(build_sequential_layer(hidden_dims_enc_common[-1], hidden_dims_enc_pre_Z[0]))
                    else:
                        mod_list_tmp.append(nn.Linear(hidden_dims_enc_common[-1], dims_z[k]))
                elif lay == self.n_layers_pre_Z:
                    mod_list_tmp.append(nn.Linear(hidden_dims_enc_pre_Z[-1],dims_z[k]))
                else:
                    mod_list_tmp.append(build_sequential_layer(hidden_dims_enc_pre_Z[lay-1],hidden_dims_enc_pre_Z[lay]))
            self.layers_latent.append(mod_list_tmp)

        
        self.sofmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
          
    def forward(self, x):
        # define the forward computation on the sample x
        
        independent_flow = [None] * self.n_inputs
        independent_flow_tmp = [None] *  self.n_inputs
        for k in range(self.n_inputs):
            for lay in range(self.n_layers_ind ):
                if lay == 0:
                    independent_flow_tmp[k] = self.layers_independent_input[k][lay](x[k])
                else:
                    independent_flow_tmp[k] = self.layers_independent_input[k][lay](independent_flow_tmp[k])
            independent_flow[k] = independent_flow_tmp[k]
        
        to_common = torch.hstack(independent_flow)



        for lay in range(self.n_layers_common):
            print(to_common.shape)
            to_common = self.layers_common[lay](to_common)
        
        latent_variables = [None] * 3
        latent_variables_tmp = [None] *  3
        for k in range(3):
            for lay in range(self.n_layers_pre_Z +1):
                if lay == 0:
                    latent_variables_tmp[k] = self.layers_latent[k][lay](to_common)
                else:
                    latent_variables_tmp[k] = self.layers_latent[k][lay](latent_variables_tmp[k])
            latent_variables[k] = latent_variables_tmp[k]
        
        

        A = self.softplus(latent_variables[0])
        B = self.softplus(latent_variables[1].t())
        Z = latent_variables[2]
       
        return A, B, Z 


# Test one input no side

INPUT_SIZE = [1000] 
Z_DIM = 30

hidden_dims_enc_ind = [512]
hidden_dims_enc_common = [256,128]
hidden_dims_enc_pre_Z = [128, 64]

enc = Encoder(INPUT_SIZE,Z_DIM, hidden_dims_enc_common, hidden_dims_enc_ind, hidden_dims_enc_pre_Z)


input = torch.ones((3000,1000))

enc([input])