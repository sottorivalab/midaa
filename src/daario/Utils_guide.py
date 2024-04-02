import pyro
import torch
import pyro.distributions.constraints as constraints
import pyro.distributions as dist

def compute_auxiliary_params_guide_input(input_types, n_dim_input, input_matrix):
    for i in pyro.plate("input_matrices", len(input_types)):
        if len(input_matrix[i].shape) > 2:
            shp_1 = len(input_matrix[i])
            input_matrix[i] = input_matrix[i].reshape([shp_1, -1])
        reconstruction_loss = 0
        with pyro.plate("data_{}".format(i), len(input_matrix[i]),dim =  -2):
            with pyro.plate("features_{}".format(i), input_matrix[i].shape[1],dim =  -1):
                if input_types[i] in ["Negative Binomial", "NB"]:
                    theta_par = pyro.param("theta_{}_par".format(i) , torch.ones(input_matrix[i].shape[1]), constraints = constraints.positive)
                    pyro.sample("theta_{}".format(i), dist.Delta(theta_par))
                elif input_types[i] in ["Gaussian", "Normal", "G", "N"]:
                    sigma_par = pyro.param("sigma_{}_par".format(i) , torch.ones(input_matrix[i].shape[1]))
                    pyro.sample("sigma_{}".format(i), dist.Delta(sigma_par))
                elif input_types[i] == "Beta":
                    N_samples_par = pyro.param("N_{}_par".format(i) , torch.ones(input_matrix[i].shape[1]) * 100, constraints = constraints.positive)
                    pyro.sample("N_{}".format(i), dist.Delta(N_samples_par))
    return None



def compute_auxiliary_params_guide_side(input_types_side, input_matrix_side, n_dim_input_side):
    for j in pyro.plate("side_matrices", len(input_types_side)):
        if len(input_matrix_side[j].shape) > 2:
            shp_1 = len(input_matrix_side[j])
            input_matrix_side[j] = input_matrix_side[j].reshape([shp_1, -1])
        reconstruction_loss = 0
        if input_types_side[j] in ["C"]:
            continue
        with pyro.plate("data_side_{}".format(j), len(input_matrix_side[j]),dim =  -2):
            with pyro.plate("features_{}_side".format(j), input_matrix_side[j].shape[1],dim =  -1):
                if input_types_side[j] in ["Negative Binomial", "NB"]:
                    theta_par = pyro.param("theta_{}_par_side".format(j) , torch.ones(len(input_types_side[j])), constraints = constraints.positive)
                    pyro.sample("theta_{}_side".format(j), dist.Delta(theta_par))
                elif input_types_side[j] in ["Gaussian", "Normal", "G", "N"]:
                    sigma_par = pyro.param("sigma_{}_par_side".format(j) , torch.ones(len(input_types_side[j])))
                    pyro.sample("sigma_{}_side".format(j), dist.Delta(torch.clamp(sigma_par, torch.tensor(0), torch.inf)))
                elif input_types_side[j] == "Beta":
                    N_samples_par = pyro.param("N_{}_par_side".format(j) , torch.ones(len(input_types_side[j])) * 100, constraints = constraints.positive)
                    pyro.sample("N_{}_side".format(j), dist.Delta(N_samples_par))
    return None


def compute_auxiliary_params_guide(
    input_types, 
    n_dim_input, 
    input_matrix, 
    input_types_side,
    input_matrix_side,
    n_dim_input_side):
    compute_auxiliary_params_guide_input(input_types, n_dim_input, input_matrix)
    if input_matrix_side is not None:
        compute_auxiliary_params_guide_side(input_types_side, input_matrix_side, n_dim_input_side)
    return None
    