import pyro
import torch
import torch.nn.functional as F
from pyro.distributions import NegativeBinomial, Normal, Poisson, OneHotCategorical, Bernoulli, Beta, Uniform, HalfNormal

def compute_loss(input_types, n_dim_input, input_reconstructed, norm_factors, input_matrix, loss_weights_reconstruction):
    reconstruction_loss_reg = 0
    reconstruction_loss_list = []
    for i in pyro.plate("input_matrices", len(input_types)):
        if len(input_matrix[i].shape) > 2:
            shp_1 = len(input_matrix[i])
            input_matrix[i] = input_matrix[i].reshape([shp_1, -1])
        reconstruction_loss = 0
        with pyro.plate("data_{}".format(i), len(input_matrix[i]),dim =  -2):
            eta = input_reconstructed[i] + torch.log(norm_factors[i]).unsqueeze(1)
            loss_type = input_types[i]
            if loss_type in ["Negative Binomial", "NB"]:
                with pyro.plate("features_{}".format(i), input_matrix[i].shape[1],dim =  -1):
                    theta = pyro.sample("theta_{}".format(i), Uniform(1e-8, 1e8))
                    reconstruction_loss += NegativeBinomial(logits=eta - torch.log(theta), 
                                                            total_count=torch.clamp(theta, 1e-9,1e9)).log_prob(input_matrix[i]).sum()
            elif loss_type in ["Gaussian", "Normal", "G", "N"]:
                with pyro.plate("features_{}".format(i), input_matrix[i].shape[1],dim =  -1):
                    sigma = pyro.sample("sigma_{}".format(i), HalfNormal(100))
                    sigma = torch.clamp(sigma, torch.tensor(1e-10), torch.inf)
                    reconstruction_loss += Normal(eta, sigma).log_prob(input_matrix[i]).sum()
            elif loss_type in ["Poisson", "P"]:
                with pyro.plate("features_{}".format(i), input_matrix[i].shape[1]):
                    reconstruction_loss += Poisson(torch.exp(eta)).log_prob(input_matrix[i]).sum()
            elif loss_type in ["Categorical", "C"]:
                reconstruction_loss += OneHotCategorical(F.softmax(eta, dim=0)).log_prob(input_matrix[i]).sum()
            elif loss_type in ["Bernoulli", "B"]:
                with pyro.plate("features_{}".format(i), input_matrix[i].shape[1],dim =  -1):
                    reconstruction_loss += Bernoulli(logits=eta).log_prob(input_matrix[i]).sum()
            elif loss_type == "Beta":
                with pyro.plate("features_{}".format(i), input_matrix[i].shape[1],dim =  -1):
                    N = pyro.sample("N_{}".format(i), Uniform(1, 100000))
                    reconstruction_loss += Beta(torch.sigmoid(eta) * N, (1 - torch.sigmoid(eta)) * N).log_prob(input_matrix[i]).sum()

        reconstruction_loss_reg += reconstruction_loss * loss_weights_reconstruction[i]
        reconstruction_loss_list.append(reconstruction_loss)
    return reconstruction_loss_reg, reconstruction_loss_list

def compute_side_loss(input_types_side, n_dim_input, side_reconstructed, input_matrix_side, loss_weights_side):
    side_loss_reg = 0
    side_loss_list = []
    for j in pyro.plate("side_matrices", len(input_types_side)):
        if len(input_matrix_side[j].shape) > 2:
            shp_1 = len(input_matrix_side[j])
            input_matrix_side[j] = input_matrix_side[j].reshape([shp_1, -1])
        side_loss = 0
        with pyro.plate("data_side_{}".format(j), len(input_matrix_side[j]), dim = -2):
            loss_type = input_types_side[j]
            if loss_type in ["Negative Binomial", "NB"]:
                with pyro.plate("features_{}_side".format(j), input_matrix_side[j].shape[1],dim =  -1):
                    theta = pyro.sample("theta_{}_side".format(j), Uniform(1e-8, 1e8)).unsqueeze(1)
                    side_loss += NegativeBinomial(logits=side_reconstructed[j] - torch.log(theta),
                                                  total_count=torch.clamp(theta, 1e-9,1e9)).log_prob(input_matrix_side[j]).sum()
            elif loss_type in ["Gaussian", "Normal", "G", "N"]:
                with pyro.plate("features_{}_side".format(j), input_matrix_side[j].shape[1],dim =  -1):
                    sigma = pyro.sample("sigma_{}_side".format(j), HalfNormal(100))
                    sigma = torch.clamp(sigma, torch.tensor(1e-10), torch.inf)
                    side_loss += Normal(side_reconstructed[j], sigma).log_prob(input_matrix_side[j]).sum()
            elif loss_type in ["Poisson", "P"]:
                with pyro.plate("features_{}_side".format(j), input_matrix_side[j].shape[1],dim =  -1):
                    side_loss += Poisson(torch.exp(side_reconstructed[j])).log_prob(input_matrix_side[j]).sum()
            elif loss_type in ["Categorical", "C"]:
                side_loss += OneHotCategorical(logits=side_reconstructed[j]).log_prob(input_matrix_side[j]).sum()
            elif loss_type in ["Bernoulli", "B"]:
                with pyro.plate("features_{}_side".format(j), input_matrix_side[j].shape[1],dim =  -1):
                    side_loss += Bernoulli(logits=side_reconstructed[j]).log_prob(input_matrix_side[j]).sum()
            elif loss_type == "Beta":
                with pyro.plate("features_{}_side".format(j), input_matrix_side[j].shape[1],dim =  -1):
                    N = pyro.sample("N_{}_side".format(j), Uniform(1, 100000)).unsqueeze(1)
                    side_loss += Beta(torch.sigmoid(side_reconstructed[j]) * N, (1- torch.sigmoid(side_reconstructed[j])) * N).log_prob(input_matrix_side[j]).sum()

        side_loss_reg += side_loss * loss_weights_side[j]
        side_loss_list.append(side_loss)

    return side_loss_reg, side_loss_list

def compute_total_loss(input_types, n_dim_input, input_reconstructed, norm_factors, input_matrix, loss_weights_reconstruction, input_types_side=None, side_reconstructed=None, input_matrix_side=None,n_dim_input_side = None, loss_weights_side=None):
    side_loss_reg = None
    side_loss_list = []
    reconstruction_loss_reg, reconstruction_loss_list = compute_loss(input_types, n_dim_input, input_reconstructed, norm_factors, input_matrix, loss_weights_reconstruction)
    total_loss = reconstruction_loss_reg
    if input_matrix_side is not None:
        side_loss_reg, side_loss_list = compute_side_loss(input_types_side, n_dim_input_side, side_reconstructed, input_matrix_side, loss_weights_side)
        total_loss += side_loss_reg
    
    return total_loss, reconstruction_loss_reg, side_loss_reg, reconstruction_loss_list, side_loss_list
