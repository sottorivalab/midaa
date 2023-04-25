import torch
import torch
import pyro
import numpy as np
import pyro.optim as optim
from pyro.infer.autoguide import AutoDelta, AutoNormal, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO,JitTrace_ELBO
from tqdm import trange


def fit_deepAA(adata,lateral_matrices = None, alpha=0.5, model_matrix = None, CUDA = True, inizialization_type = "best_ELBO_random", torch_seed = 3):

    if CUDA and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type(t=torch.FloatTensor)


    torch.manual_seed(torch_seed)
    pyro.clear_param_store()

    input_matrix = torch.tensor(adata.X)
    model_matrix  =  model_matrix
    UMI = torch.tensor(adata.n_counts / np.mean(adata.n_counts))
    lateral_matrices = [ lateral_matrices]

    deepAA = DeepAA(input_matrix.shape[0], input_matrix_atlas.shape[1], input_matrix_hallmarks.shape[0], narchetypes=11, batch_size=37000)
    steps = 2000
    t = trange(steps, desc='Bar desc', leave = True)
    loss = Trace_ELBO
    svi = SVI(deepAA.model, deepAA.guide, optim.ClippedAdam({"lr": 0.005}), loss=loss())
    elbo_list = [] 

    for j in t:
        elbo = svi.step(input_matrix, model_matrix, UMI, lateral_matrices, write_log = True )
        elbo_list.append(elbo )
        _ = t.set_description('ELBO: {:.5f}  '.format(elbo ))
        _ = t.refresh()

    A,B = deepAA.encoder(input_matrix.t())

    archetypes_inferred = B @ input_matrix.t()

    Z_fixed = deepAA.Z_fix

    theta = pyro.param("theta_param") 

    beta_mean = pyro.param("beta_means_param") 
    beta_cov = pyro.param("beta_cov_param") 

    eta_1 = torch.matmul(model_matrix, beta_mean.t())

    eta_2, hallmarks_mu, class_probability = deepAA.decoder(A @ deepAA.Z_fix)

    eta = eta_1 + eta_2 + torch.log(UMI).unsqueeze(1)

    mu = eta.exp()

    r_loss = deepAA.model(input_matrix, model_matrix, UMI, lateral_matrices)

    params = {
    "beta_mean" : beta_mean.cpu().detach().numpy(),
    "beta_cov" : beta_cov.cpu().detach().numpy(),
    "theta" : theta.cpu().detach().numpy(),
    "A" : A.cpu().detach().numpy(),
    "B" : B.cpu().detach().numpy(),
    "Z" : archetypes_inferred.cpu().detach().numpy(),
    "Z_fixed" : Z_fixed.cpu().detach().numpy(),
    "eta_1" : eta_1.cpu().detach().numpy(),
    "eta_2" : eta_2.cpu().detach().numpy(),
    "eta" : eta.cpu().detach().numpy(),
    "mu" : mu.cpu().detach().numpy()
    }
