from Encoder import *
from Decoder import *
from Utils import *

class DeepAA(nn.Module):
  
  def __init__(self,n_genes, n_celltype, n_hallmarks, prior_loc = 10, batch_size = 5120, 
  decoder_hidden_dims = [512,256,128],
  encoder_hidden_dims = [512,256,128],
  theta_bounds = (1e-5, 10000),
  init_loc = 0.1, init_theta = 1,  alpha = 3,
  narchetypes = 10):
    
    super().__init__()
    
    self.prior_loc = prior_loc
    self.decoder_hidden_dims = decoder_hidden_dims
    self.encoder_hidden_dims = encoder_hidden_dims
    self.theta_bounds = theta_bounds
    self.init_loc = init_loc
    self.init_theta = init_theta
    self.narchetypes = narchetypes
    self.Z_fix = create_z_fix(self.narchetypes - 1)
    self.batch_size = batch_size
    
    self.encoder = Encoder(n_genes, narchetypes - 1,decoder_hidden_dims, False)
    self.decoder = Decoder(n_genes, narchetypes - 1 ,decoder_hidden_dims, False, ["gaussian", "categorical"], [n_hallmarks, n_celltype] ,True )
    
  def model(self,input_matrix, model_matrix, UMI,
  input_matrix_hallmarks, input_matrix_atlas, write_log = False):
    
    n_cells = input_matrix.shape[1]
    n_genes = input_matrix.shape[0]
    n_features = model_matrix.shape[1]
    n_celltype = input_matrix_atlas.shape[1]
    n_hallmarks = input_matrix_hallmarks.shape[0]
    
    pyro.module("decoder",self.decoder )
    ### Library size factors ###
    z_loc_predicted = torch.zeros(self.narchetypes - 1)
    z_scale = torch.ones(self.narchetypes - 1) 
    z_loc = torch.zeros(self.narchetypes - 1)
    
    with pyro.plate("genes", n_genes):
      theta = pyro.sample("theta", dist.Uniform(self.theta_bounds[0],self.theta_bounds[1])).unsqueeze(0)
      with pyro.plate("features", n_features):
        beta_prior_mu =  pyro.sample("beta_prior_mu", dist.Laplace(0,1))
      beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu.t(), scale_tril=torch.eye(n_features, n_features) * self.prior_loc, validate_args=False))
    if self.batch_size < n_cells:
      with pyro.plate("data", n_cells, subsample_size= self.batch_size) as ind:
        eta_1 = torch.matmul(model_matrix[ind,:], beta.t())
        latent_var = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1) )
        eta_2, hallmarks_mu, class_probability = self.decoder(latent_var)
        
        eta = eta_1 + eta_2 + torch.log(UMI[ind]).unsqueeze(1)
        
        reconstruction_loss = dist.NegativeBinomial(logits = eta - torch.log(theta),
         total_count= torch.clamp(theta, 1e-9,1e9)).log_prob(input_matrix[:,ind].t()).sum()
        cell_type_loss = dist.OneHotCategorical(class_probability).log_prob(input_matrix_atlas[ind,:]).sum()
        archetype_loss = dist.Normal(hallmarks_mu,1).log_prob(input_matrix_hallmarks[:,ind].t()).sum()
    else:
      with pyro.plate("data", n_cells):
        eta_1 = torch.matmul(model_matrix, beta.t())
        latent_var = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1) )
        eta_2, hallmarks_mu, class_probability = self.decoder(latent_var)
        
        eta = eta_1 + eta_2 + torch.log(UMI).unsqueeze(1)
        
        reconstruction_loss = dist.NegativeBinomial(logits = eta - torch.log(theta),
         total_count= torch.clamp(theta, 1e-9,1e9)).log_prob(input_matrix.t()).sum()
        cell_type_loss = dist.OneHotCategorical(class_probability).log_prob(input_matrix_atlas).sum()
        archetype_loss = dist.Normal(hallmarks_mu,1).log_prob(input_matrix_hallmarks.t()).sum()
      
    with pyro.plate("ndims", self.narchetypes - 1):
      with pyro.plate("archs", self.narchetypes):
        Z_predicted = pyro.sample("Z_predicted", dist.Uniform(torch.ones([self.narchetypes, self.narchetypes -1  ]) * -100,torch.ones([self.narchetypes, self.narchetypes -1  ]) * 100))
    
    Z_loss =torch.nn.functional.mse_loss(Z_predicted, self.Z_fix, reduction='sum')
    pyro.factor("loss", reconstruction_loss + 
    cell_type_loss * n_genes + 
    archetype_loss * (n_genes / n_hallmarks) - 
    Z_loss * (self.batch_size * n_genes) )
    
    if write_log:
      reconstruction_loss_sr = reconstruction_loss.clone().detach().cpu().numpy()
      Z_loss_sr = (Z_loss * (self.batch_size * n_genes) / (self.narchetypes * (self.narchetypes - 1)) * 10 ).clone().detach().cpu().numpy() * -1
      archetype_loss_sr = (archetype_loss * (n_genes / n_hallmarks)).clone().detach().cpu().numpy()
      cell_type_loss_sr = (cell_type_loss * (n_genes / n_celltype)).clone().detach().cpu().numpy()
      with open("logfile.txt", "a") as file1:
        file1.write(str(reconstruction_loss_sr) + "\t" + str(Z_loss_sr) + "\t" + str(archetype_loss_sr) + "\t" + str(cell_type_loss_sr) + "\n")
    
    return reconstruction_loss
    
    
  def guide(self, input_matrix, model_matrix, UMI,
  input_matrix_hallmarks, input_matrix_atlas, write_log = False, batch = True):
      
      n_cells = input_matrix.shape[1]
      n_genes = input_matrix.shape[0]
      n_features = model_matrix.shape[1]
      n_celltype = input_matrix_atlas.shape[1]
      n_hallmarks = input_matrix_hallmarks.shape[0]
      
      pyro.module("encoder", self.encoder)
      self.Z_fix = create_z_fix(self.narchetypes - 1)
      
      beta_means = pyro.param("beta_means_param", torch.zeros((n_genes, n_features)))
      beta_cov = pyro.param("beta_cov_param", torch.eye(n_features, n_features).repeat([n_genes,1,1]) * self.prior_loc, constraint=constraints.lower_cholesky)
      theta_param = pyro.param("theta_param",init_tensor= torch.ones(n_genes)* self.init_theta,  constraint=constraints.positive)
      with pyro.plate("genes", n_genes):
        theta = pyro.sample("theta", dist.Delta(theta_param))
        with pyro.plate("features", n_features):
          beta_prior_mu =  pyro.sample("beta_prior_mu",dist.Delta(beta_means.t()))
        beta = pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu.t(), scale_tril=beta_cov, validate_args=False))
      if self.batch_size < n_cells:
        with pyro.plate("data", n_cells, subsample_size= self.batch_size) as ind:
            A, B = self.encoder(input_matrix[:,ind].t())
            z_loc = A @ self.Z_fix
            Z_predicted = B @ z_loc
            pyro.sample("latent", dist.Delta(z_loc).to_event(1))
        with pyro.plate("ndims", self.narchetypes - 1):
           with pyro.plate("archs", self.narchetypes):
            pyro.sample("Z_predicted", dist.Delta(Z_predicted))
      else:
        with pyro.plate("data", n_cells):
            A, B = self.encoder(input_matrix.t())
            z_loc = A @ self.Z_fix
            Z_predicted = B @ z_loc
            pyro.sample("latent", dist.Delta(z_loc).to_event(1))
        with pyro.plate("ndims", self.narchetypes - 1):
           with pyro.plate("archs", self.narchetypes):
            pyro.sample("Z_predicted", dist.Delta(Z_predicted))
