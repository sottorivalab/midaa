import numpy as np


def calculate_BIC(res):
    pass



def log_lk(deepAA,input_matrix, UMI, model_matrix):
    eta_1 = torch.matmul(model_matrix, pyro.param("beta_means_param").t())
    A,B = deepAA.encoder(input_matrix.t())
    z_loc = A @ deepAA.Z_fix
    eta_2, hallmarks_mu, class_probability = deepAA.decoder(z_loc)
    
    eta = eta_1 * 0 + eta_2 + torch.log(UMI).unsqueeze(1)
    
    reconstruction_loss = dist.NegativeBinomial(logits = eta - torch.log(pyro.param("theta_param") ),
     total_count= torch.clamp(pyro.param("theta_param") , 1e-9,1e9)).log_prob(input_matrix.t()).sum()
    return(reconstruction_loss)

def retrieve_params():
    param_names = pyro.get_param_store().keys()
    res = {nms: pyro.param(nms) for nms in param_names}
    return res
 
def calculate_number_of_params(params):
    res = 0
    for i in params:
         res += torch.prod(torch.tensor(params[i].shape))
    return res

def calculate_BIC(res):
    lkk = log_lk(res)
    n_pars = calculate_number_of_params(retrieve_params())
    n_samples = input_matrix.shape[1]
    BIC = -2 * lkk + n_pars * torch.log(torch.tensor(n_samples))
    return(-1 * lkk)