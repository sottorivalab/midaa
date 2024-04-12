import torch
import numpy as np
from torch.utils.data import Dataset


class MultiOmeDataset(Dataset):
    def __init__(self, input_data, side_data = None,model_matrix = None, cuda = False, load_all_on_gpu = True, from_file = None):
        
         ### Get information of shape ###
        self.n_inputs = len(input_matrix)

        self.n_samples_input = [mat.shape[0] for mat in input_matrix]
        self.n_features = [mat.shape[1] for mat in input_matrix]
        if model_matrix is not None:
            self.n_covariates = model_matrix.shape[1]

        if input_matrix_side is not None:
            
            self.n_side = len(input_matrix_side)
            self.n_samples_side = [mat.shape[0] for mat in input_matrix_side]
            self.n_features_side = [mat.shape[1] for mat in input_matrix_side]

            if loss_weights_side is None:
                loss_weights_side = [1/torch.tensor(n_features_side[i]/n_features_side[0]) for i in range(n_side)]
            else:
                loss_weights_side = [torch.tensor(loss_weights_side[i]) for i in range(n_side)]
        
        self.counts = torch.tensor(self.data).unsqueeze(0).repeat(n_gs, 1, 1)
        self.genes = adata.var_names
        self.cells = adata.obs_names
        self.n_genes = adata.n_vars
        self.n_cells = adata.n_obs

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        return self.counts[:,idx,:]