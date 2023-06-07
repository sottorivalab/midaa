import pyro
import torch
import torch.nn.functional as F
import pyro.distributions.constraints as constraints
import pyro.distributions as dist

from scdeepaa.Encoder import *
from scdeepaa.Decoder import *
from scdeepaa.Utils import *

class DeepAA(nn.Module):
  
  def __init__(self,n_dim_input, input_types,
  n_dim_input_side = None, 
  input_types_side = None,
  batch_size = 5120, 
  hidden_dims_dec_common = [128,256],
  hidden_dims_dec_last = [1024],
  hidden_dims_enc_ind = [512],
  hidden_dims_enc_common = [256,128],
  hidden_dims_enc_pre_Z = [128, 64],
  theta_bounds = (1e-5, 10000),
  init_loc = 0.1, init_theta = 1,  alpha = 3,
  prior_loc = 10,
  narchetypes = 10,
  fix_Z = False,
  Z_fix_norm = None):
    
    super().__init__()
    
    ### Input values ###
    
    self.n_dim_input = n_dim_input
    self.input_types = input_types
    self.n_dim_input_side = n_dim_input_side
    self.input_types_side = input_types_side

    ### Decoder attributes ###
    self.hidden_dims_dec_common = hidden_dims_dec_common
    self.hidden_dims_dec_last = hidden_dims_dec_last
    
    ### Encoder attributes ###
    self.hidden_dims_enc_ind = hidden_dims_enc_ind
    self.hidden_dims_enc_common = hidden_dims_enc_common
    self.hidden_dims_enc_pre_Z = hidden_dims_enc_pre_Z
    
    ### Parameters distribution ###
    self.prior_loc = prior_loc
    self.theta_bounds = theta_bounds
    self.init_loc = init_loc
    self.init_theta = init_theta
    
    ### Other parameters ###
    self.narchetypes = narchetypes
    self.latent_dims = narchetypes -1
    self.fix_Z = fix_Z
    self.Z_fix = create_z_fix(self.narchetypes - 1)
    self.Z_fix_norm = Z_fix_norm
    self.batch_size = batch_size
    
    ### Init Encoder and Decoder ###
    self.encoder = Encoder(self.n_dim_input, self.narchetypes - 1, self.hidden_dims_enc_common, 
                           self.hidden_dims_enc_ind,self.hidden_dims_enc_pre_Z, fix_Z = self.fix_Z )
    self.decoder = Decoder(self.n_dim_input, self.narchetypes - 1 ,
                           self.hidden_dims_dec_common, self.hidden_dims_dec_last,
                             self.input_types, self.n_dim_input_side, self.input_types_side)
    
  def model(
        self,input_matrix, model_matrix, 
        norm_factors,input_matrix_side, 
        loss_weights_reconstruction = None, 
        loss_weights_side = None
        ):
    

    ### Get information of shape ###

    n_inputs = len(input_matrix)
    
    n_samples_input = [mat.shape[0] for mat in input_matrix]
    n_features = [mat.shape[1] for mat in input_matrix]
    if model_matrix is not None:
        n_covariates = model_matrix.shape[1]

    if input_matrix_side is not None:
        n_side = len(input_matrix_side)
        n_samples_side = [mat.shape[0] for mat in input_matrix_side]
        n_features_side = [mat.shape[1] for mat in input_matrix_side]

        if loss_weights_side is None:
            loss_weights_side = [torch.tensor(n_features_side[i]/n_features_side[0]) for i in range(n_side)]
        else:
            loss_weights_side = [torch.tensor(loss_weights_side[i]) for i in range(n_side)]
    

    
    if loss_weights_reconstruction is None:
        loss_weights_reconstruction = [torch.tensor(n_features[i]/n_features[0]) for i in range(n_inputs)]
    else:
        loss_weights_reconstruction = [torch.tensor(loss_weights_reconstruction[i]) for i in range(n_inputs)]

    
    
    pyro.module("decoder",self.decoder)

    if model_matrix is not None:
        with pyro.plate("features", n_covariates):
            beta_prior_mu =  pyro.sample("beta_prior_mu", dist.Laplace(0,1))

        beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu.t(), 
                                                        scale_tril=torch.eye(n_features, n_features) * self.prior_loc, 
                                                        validate_args=False))
        eta_regr = torch.matmul(model_matrix, beta.t())
    
    with pyro.plate("narchetypes", self.narchetypes):
        z_scale = torch.ones(self.latent_dims) 
        z_loc = torch.zeros(self.latent_dims)
        latent_var_Z = pyro.sample("latent_Z", dist.Normal(z_loc, z_scale).to_event(1))
        
    with pyro.plate("all_samples", n_samples_input[0]):
        z_scale = torch.ones(self.latent_dims) 
        z_loc = torch.zeros(self.latent_dims)
        concentration_dir = torch.ones(self.narchetypes) 
        
        latent_var_A = pyro.sample("latent_A", dist.Dirichlet(concentration_dir))
        latent_var = latent_var_A @ latent_var_Z
        if model_matrix is not None:
            input_reconstructed, side_reconstructed = self.decoder(eta_regr + latent_var)
        else:
            input_reconstructed, side_reconstructed = self.decoder(latent_var)

    reconstruction_loss_reg = 0
    reconstruction_loss = 0
    for i in pyro.plate("input_matrices", n_inputs):

        reconstruction_loss = 0
        with pyro.plate("data_{}".format(i), n_samples_input[i]):
            
            eta = input_reconstructed[i][:,:self.n_dim_input[i]] + torch.log(norm_factors[i]).unsqueeze(1)

            if self.input_types[i] == "Negative Binomial" or self.input_types[i] == "NB":
                reconstruction_loss += dist.NegativeBinomial(logits = eta - torch.log(input_reconstructed[i][:,self.n_dim_input[i]:]),
                    total_count= torch.clamp(torch.exp(input_reconstructed[i][:,self.n_dim_input[i]:]), 1e-9,1e9)).log_prob(input_matrix[i]).sum()
            if self.input_types[i] == "Gaussian" or self.input_types[i] == "Normal" or self.input_types[i] == "G" or self.input_types[i] == "N" :
                #reconstruction_loss += dist.Normal(eta , F.softplus(input_reconstructed[i][:,self.n_dim_input[i]:])).log_prob(input_matrix[i]).sum()
                reconstruction_loss += dist.Normal(eta , .1).log_prob(input_matrix[i]).sum()
            if self.input_types[i] == "Poisson" or self.input_types[i] == "P":
                reconstruction_loss += dist.Poisson(torch.exp(eta)).log_prob(input_matrix[i]).sum()
            if self.input_types[i] == "Categorical" or self.input_types[i] == "C":
                reconstruction_loss += dist.OneHotCategorical(F.softmax(eta, dim = 0)).log_prob(input_matrix[i]).sum()
            if self.input_types[i] == "Bernoulli" or self.input_types[i] == "B":
                reconstruction_loss += dist.Bernoulli(logits = eta).log_prob(input_matrix[i]).sum()

            reconstruction_loss_reg += reconstruction_loss * loss_weights_reconstruction[i]
                
    
    side_loss_reg = 0
    side_loss = 0
    if input_matrix_side is not None:
        for j in pyro.plate("side_matrices", n_side):
            side_loss = 0
            with pyro.plate("data_side_{}".format(j), n_samples_side[j]):
                if self.input_types_side[j] == "Negative Binomial" or self.input_types_side[j] == "NB":
                    side_loss += dist.NegativeBinomial(logits = side_reconstructed[j][:self.n_dim_input[j]] - torch.log(side_reconstructed[j][self.n_dim_input[j]:]),
                        total_count= torch.clamp(F.softplus(side_reconstructed[j][1]), 1e-9,1e9)).log_prob(input_matrix_side[j]).sum()
                if self.input_types_side[j] == "Gaussian" or self.input_types_side[j] == "Normal" or self.input_types_side[j] == "G" or self.input_types_side[j] == "N" :
                    side_loss += dist.Normal(side_reconstructed[j][:self.n_dim_input[j]] , F.softplus(side_reconstructed[j][self.n_dim_input[j]:])).log_prob(input_matrix_side[j]).sum()
                if self.input_types_side[j] == "Poisson" or self.input_types_side[j] == "P":
                    side_loss += dist.Poisson(torch.exp(side_reconstructed[j])).log_prob(input_matrix_side[j]).sum()
                if self.input_types_side[j] == "Categorical" or self.input_types_side[j] == "C":
                    side_loss += dist.OneHotCategorical(logits = side_reconstructed[j]).log_prob(input_matrix_side[j]).sum()
                if self.input_types_side[j] == "Bernoulli" or self.input_types_side[j] == "B":
                    side_loss += dist.Bernoulli(logits = side_reconstructed[j]).log_prob(input_matrix_side[j]).sum()
                
            side_loss_reg += side_loss * loss_weights_side[j]

            

    if self.fix_Z:
        if self.Z_fix_norm is None:
            self.Z_fix_norm = n_features[0] * n_inputs
        with pyro.plate("ndims", self.narchetypes - 1):
            with pyro.plate("archs", self.narchetypes ):
                Z_predicted = pyro.sample("Z_predicted", dist.Uniform(-1000, 1000))
        
        Z_loss = torch.nn.functional.mse_loss(Z_predicted, self.Z_fix, reduction='sum') * self.Z_fix_norm
        pyro.factor("loss", reconstruction_loss_reg + side_loss_reg + Z_loss)
    else:
        pyro.factor("loss", reconstruction_loss_reg + side_loss_reg)
        
    """         
    if write_log:
        reconstruction_loss_sr = reconstruction_loss.clone().detach().cpu().numpy()
        Z_loss_sr = (Z_loss * (self.batch_size * n_genes) / (self.narchetypes * (self.narchetypes - 1)) * 10 ).clone().detach().cpu().numpy() * -1
        archetype_loss_sr = (archetype_loss * (n_genes / n_hallmarks)).clone().detach().cpu().numpy()
        cell_type_loss_sr = (cell_type_loss * (n_genes / n_celltype)).clone().detach().cpu().numpy()
        with open("logfile.txt", "a") as file1:
            file1.write(str(reconstruction_loss_sr) + "\t" + str(Z_loss_sr) + "\t" + str(archetype_loss_sr) + "\t" + str(cell_type_loss_sr) + "\n") 
    """
        
    return reconstruction_loss_reg,side_loss_reg,loss_weights_reconstruction, loss_weights_side,reconstruction_loss, side_loss, reconstruction_loss_reg + side_loss_reg
        
        
  def guide( self,input_matrix, model_matrix, 
        norm_factors,input_matrix_side, 
        loss_weights_reconstruction = None, 
        loss_weights_side = None):
      
    n_samples_input = [mat.shape[0] for mat in input_matrix]
    n_features = [mat.shape[1] for mat in input_matrix]
    if model_matrix is not None:
        n_covariates = model_matrix.shape[1]
        
    pyro.module("encoder", self.encoder)
    #self.Z_fix = create_z_fix(self.narchetypes - 1)
    if model_matrix is not None:
        beta_means = pyro.param("beta_means_param", torch.zeros((self.latent_dims, n_covariates)))
        beta_cov = pyro.param("beta_cov_param", torch.eye(n_covariates, n_covariates).repeat([self.latent_dims,1,1]) * self.prior_loc, constraint=constraints.lower_cholesky)
        with pyro.plate("features", n_features):
            beta_prior_mu =  pyro.sample("beta_prior_mu",dist.Delta(beta_means.t()))
            pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu.t(), scale_tril=beta_cov, validate_args=False))

    A, B, Z = self.encoder(input_matrix)
    if self.fix_Z:
        Z_predicted = Z
        with pyro.plate("ndims", self.narchetypes - 1):
            with pyro.plate("archs", self.narchetypes):
                pyro.sample("Z_predicted", dist.Delta(B @ A @ Z))
    else:
        Z_predicted = B @ Z
        
    with pyro.plate("narchetypes", self.narchetypes):
        pyro.sample("latent_Z", dist.Delta(Z_predicted).to_event(1))
    with pyro.plate("all_samples", n_samples_input[0]):
        pyro.sample("latent_A", dist.Delta(torch.clamp(A, 1e-20,1)).to_event(1))


