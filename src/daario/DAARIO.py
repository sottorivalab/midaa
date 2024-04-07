import pyro
import torch
import torch.nn.functional as F
import pyro.distributions.constraints as constraints
import pyro.distributions as dist

from daario.Encoder import *
from daario.Decoder import *
from daario.Utils import *
from daario.Likelihood import *
from daario.Utils_guide import compute_auxiliary_params_guide


class DAARIO(nn.Module):
    def __init__(self,
        n_dim_input, 
        input_types,
        n_dim_input_side = None, 
        input_types_side = None,
        batch_size = 5120, 
        hidden_dims_dec_common = [128,256],
        hidden_dims_dec_last = [1024],
        hidden_dims_dec_last_side = None,
        hidden_dims_enc_ind = [512],
        hidden_dims_enc_common = [256,128],
        hidden_dims_enc_pre_Z = [128, 64],
        layers_independent_types = None,
        layers_independent_types_side = None,
        output_types_side = None,
        image_size = [256,256],
        theta_bounds = (1e-5, 10000),
        init_loc = 0.1, init_theta = 1,  alpha = 3,
        prior_loc = 10,
        narchetypes = 10,
        fix_Z = False,
        Z_fix_norm = None,
        Z_fix_release = False,
        initialization_mode_step1 = False,
        initialization_mode_step2 = False,
        just_VAE = False,
        linearize_encoder = False,
        linearize_decoder = False,
        kernel_size=3, 
        stride=1, 
        padding=1, 
        pool_size=2, 
        pool_stride=2):
        
        super().__init__()

        ### Input values ###

        self.n_dim_input = n_dim_input
        self.input_types = input_types
        self.n_dim_input_side = n_dim_input_side
        self.input_types_side = input_types_side

        ### Decoder attributes ###
        self.hidden_dims_dec_common = hidden_dims_dec_common
        self.hidden_dims_dec_last = hidden_dims_dec_last
        self.hidden_dims_dec_last_side = hidden_dims_dec_last_side
        self.layers_independent_types_side = layers_independent_types_side

        ### Encoder attributes ###
        self.hidden_dims_enc_ind = hidden_dims_enc_ind
        self.hidden_dims_enc_common = hidden_dims_enc_common
        self.hidden_dims_enc_pre_Z = hidden_dims_enc_pre_Z
        self.layers_independent_types = layers_independent_types

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
        self.Z_fix_release = Z_fix_release
        self.batch_size = batch_size
        self.just_VAE = just_VAE
        self.linearize_encoder = linearize_encoder
        self.linearize_decoder = linearize_decoder

        ### Parameters for convolutional ###
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        ### Initialization Parameters ###

        self.initialization_mode_step1 = initialization_mode_step1
        self.initialization_mode_step2 = initialization_mode_step2

        ### Init Encoder and Decoder ###
        self.encoder = Encoder(input_size = self.n_dim_input, 
                                z_dim = self.narchetypes - 1, 
                                hidden_dims_enc_common = self.hidden_dims_enc_common, 
                                hidden_dims_enc_ind = self.hidden_dims_enc_ind,
                                hidden_dims_enc_pre_Z = self.hidden_dims_enc_pre_Z, 
                                fix_Z = self.fix_Z,
                                linearize = self.linearize_encoder,
                                layers_independent_types = self.layers_independent_types,
                                image_size = self.image_size,
                                kernel_size = self.kernel_size,
                                stride = self.stride,
                                padding = self.padding,
                                pool_size = self.pool_size,
                                pool_stride = self.pool_stride)

        self.decoder = Decoder(input_size = self.n_dim_input, 
                               z_dim = self.narchetypes - 1,
                               hidden_dims_dec_common = self.hidden_dims_dec_common, 
                               hidden_dims_dec_last = self.hidden_dims_dec_last,
                               output_types_input = self.input_types, 
                               layers_independent_types = self.layers_independent_types,
                               input_size_aux = self.n_dim_input_side, 
                               output_types_side = self.input_types_side,
                               hidden_dims_dec_last_side = self.hidden_dims_dec_last_side,
                               layers_independent_types_side = self.layers_independent_types_side,
                               linearize = self.linearize_decoder, 
                               image_size = self.image_size, 
                               kernel_size=self.kernel_size, 
                               stride=self.stride, 
                               padding=self.padding,
                               pool_size=self.pool_size)

    def model(
        self,input_matrix, model_matrix, 
        norm_factors,input_matrix_side, 
        loss_weights_reconstruction = None, 
        loss_weights_side = None, 
        initialization_input = None,
        initialization_B_weight = None):

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
                loss_weights_side = [1/torch.tensor(n_features_side[i]/n_features_side[0]) for i in range(n_side)]
            else:
                loss_weights_side = [torch.tensor(loss_weights_side[i]) for i in range(n_side)]



        if loss_weights_reconstruction is None:
            loss_weights_reconstruction = [1/torch.tensor(n_features[i]/n_features[0]) for i in range(n_inputs)]
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

        if self.fix_Z:
            if self.Z_fix_norm is None:
                self.Z_fix_norm = n_features[0] * n_inputs
            with pyro.plate("ndims", self.narchetypes - 1):
                with pyro.plate("archs", self.narchetypes ):
                    Z_predicted = pyro.sample("Z_predicted", dist.Uniform(-1000, 1000))

        if self.initialization_mode_step2:

            with pyro.plate("narchetypes", self.narchetypes):
                #z_scale = torch.ones(self.latent_dims) 
                #z_loc = torch.zeros(self.latent_dims)
                #latent_var_Z = pyro.sample("latent_Z", dist.Normal(z_loc, z_scale).to_event(1))
                latent_var_B = pyro.sample("latent_B", dist.Dirichlet(torch.ones(n_samples_input[0])))
                if self.fix_Z or self.linearize_decoder:
                        z_scale = torch.ones(self.latent_dims) 
                        z_loc = torch.zeros(self.latent_dims)
                        latent_var_Z = pyro.sample("latent_Z", dist.Normal(z_loc, z_scale).to_event(1))
                else:
                    U = torch.ones(self.latent_dims) + 0.1
                    L = torch.ones(self.latent_dims) * -1.1
                    latent_var_Z = pyro.sample("latent_Z", dist.Uniform(L, U).to_event(1))


            with pyro.plate("all_samples", n_samples_input[0]):
                concentration_dir = torch.ones(self.narchetypes) 
                latent_var_A = pyro.sample("latent_A", dist.Dirichlet(concentration_dir))
                latent_var = latent_var_A @ latent_var_Z
                if model_matrix is not None:
                    input_reconstructed, side_reconstructed = self.decoder(eta_regr + latent_var)
                else:
                    input_reconstructed, side_reconstructed = self.decoder(latent_var)
            
            B_loss = torch.nn.functional.mse_loss(latent_var_B,initialization_input["B"], reduction='sum')
            #print(((initialization_input["B"] - latent_var_B)**2).sum())
            #print(((initialization_input["B"] - latent_var_B)**2).sum())
            #print(B_loss)
            pyro.factor("B_loss", -B_loss * initialization_B_weight )


        else:
            if not self.just_VAE:
                with pyro.plate("narchetypes", self.narchetypes):
                    #z_scale = torch.ones(self.latent_dims) 
                    #z_loc = torch.zeros(self.latent_dims)
                    #latent_var_Z = pyro.sample("latent_Z", dist.Normal(z_loc, z_scale).to_event(1))
                    if self.fix_Z or self.linearize_decoder or self.linearize_encoder:
                        z_scale = torch.ones(self.latent_dims) 
                        z_loc = torch.zeros(self.latent_dims)
                        latent_var_Z = pyro.sample("latent_Z", dist.Normal(z_loc, z_scale).to_event(1))
                    else:
                        U = torch.ones(self.latent_dims) + 0.1
                        L = torch.ones(self.latent_dims) * -1.1
                        latent_var_Z = pyro.sample("latent_Z", dist.Uniform(L, U).to_event(1))


            with pyro.plate("all_samples", n_samples_input[0]):
                if not self.just_VAE:
                    concentration_dir = torch.ones(self.narchetypes) 
                    latent_var_A = pyro.sample("latent_A", dist.Dirichlet(concentration_dir))
                    latent_var = latent_var_A @ latent_var_Z
                else:
                    z_scale = torch.ones(self.latent_dims) 
                    z_loc = torch.zeros(self.latent_dims)
                    latent_var_Z = pyro.sample("latent_Z", dist.Normal(z_loc, z_scale).to_event(1))
                    latent_var = latent_var_Z
                if model_matrix is not None:
                    input_reconstructed, side_reconstructed = self.decoder(eta_regr + latent_var)
                else:
                    input_reconstructed, side_reconstructed = self.decoder(latent_var)

        # Assuming the relevant variables have been initialized (e.g., input_types, n_dim_input, etc.)
        if not self.initialization_mode_step2:
            total_loss, reconstruction_loss_reg, side_loss_reg, reconstruction_loss, side_loss = compute_total_loss(
                input_types=self.input_types, 
                n_dim_input=self.n_dim_input, 
                input_reconstructed=input_reconstructed, 
                norm_factors=norm_factors, 
                input_matrix=input_matrix, 
                loss_weights_reconstruction=loss_weights_reconstruction,
                input_types_side=self.input_types_side,
                side_reconstructed=side_reconstructed,
                input_matrix_side=input_matrix_side,
                n_dim_input_side = self.n_dim_input_side,
                loss_weights_side=loss_weights_side
            )

            if self.fix_Z:
                Z_loss = torch.nn.functional.mse_loss(Z_predicted, self.Z_fix, reduction='sum') 
                Z_loss_reg = Z_loss * self.Z_fix_norm
                pyro.factor("loss", total_loss - Z_loss_reg)
                return reconstruction_loss_reg,side_loss_reg,loss_weights_reconstruction, loss_weights_side,reconstruction_loss, side_loss, total_loss, Z_loss,  Z_loss_reg
            else:
                pyro.factor("loss", total_loss )
                return reconstruction_loss_reg,side_loss_reg,loss_weights_reconstruction, loss_weights_side,reconstruction_loss, side_loss, total_loss






    def guide( self,input_matrix, model_matrix, 
        norm_factors,input_matrix_side, 
        loss_weights_reconstruction = None, 
        loss_weights_side = None, 
        initialization_input = None,
        initialization_B_weight = None):

        n_samples_input = [mat.shape[0] for mat in input_matrix]
        n_features = [mat.shape[1] for mat in input_matrix]
        if model_matrix is not None:
            n_covariates = model_matrix.shape[1]

        pyro.module("encoder", self.encoder)
        
        if model_matrix is not None:
            beta_means = pyro.param("beta_means_param", torch.zeros((self.latent_dims, n_covariates)))
            beta_cov = pyro.param("beta_cov_param", torch.eye(n_covariates, n_covariates).repeat([self.latent_dims,1,1]) * self.prior_loc, constraint=constraints.lower_cholesky)
            with pyro.plate("features", n_features):
                beta_prior_mu =  pyro.sample("beta_prior_mu",dist.Delta(beta_means.t()))
                pyro.sample("beta", dist.MultivariateNormal(beta_prior_mu.t(), scale_tril=beta_cov, validate_args=False))

        A, B_out, Z = self.encoder(input_matrix)
        if self.just_VAE:
            with pyro.plate("all_samples", n_samples_input[0]):
                pyro.sample("latent_Z", dist.Delta(Z).to_event(1))
        else:
            
            if self.initialization_mode_step1:
                B = initialization_input["B"]
            else:
                B = torch.clamp(B_out, 1e-15, 1 - 1e-15)
                
            if self.fix_Z:
                if self.Z_fix_release:
                    latent_Z_params = pyro.param("archetypes_params", Z)
                else:
                    latent_Z_params = Z
                with pyro.plate("ndims", self.narchetypes - 1):
                    with pyro.plate("archs", self.narchetypes):
                        pyro.sample("Z_predicted", dist.Delta(B @ A @ latent_Z_params))
            else:
                latent_Z_params = B @ Z

            with pyro.plate("narchetypes", self.narchetypes):
                pyro.sample("latent_Z", dist.Delta(latent_Z_params).to_event(1))
                if self.initialization_mode_step2:
                    B_out = torch.clamp(B_out, 1e-14, 1 - 1e-14)
                    B_out = F.normalize(B_out, p=1, dim=1)
                    pyro.sample("latent_B", dist.Delta(B_out).to_event(1))
            with pyro.plate("all_samples", n_samples_input[0]):
                pyro.sample("latent_A", dist.Delta(torch.clamp(A, 1e-20,1)).to_event(1))

        if not self.initialization_mode_step2:
            _ = compute_auxiliary_params_guide(
                input_types=self.input_types, 
                n_dim_input=self.n_dim_input, 
                input_matrix=input_matrix, 
                input_types_side=self.input_types_side,
                input_matrix_side=input_matrix_side,
                n_dim_input_side = self.n_dim_input_side
            )



