import torch
import torch
import pyro
import numpy as np
import pyro.optim as optim
from pyro.infer.autoguide import AutoDelta, AutoNormal, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO,JitTrace_ELBO
from tqdm import trange


from scdeepaa.Utils import sample_batch, to_cpu_ot_iterate, detach_or_iterate
from scdeepaa.deepAA import DeepAA

def fit_deepAA(input_matrix,
               normalization_factor = None,
               input_types = ["NB"],
               loss_weights_reconstruction = None,
               side_matrices = None, 
               input_types_side = None,
               loss_weights_side = None,
               hidden_dims_dec_common = [256,512],
               hidden_dims_dec_last = [1024],
               hidden_dims_enc_ind = [1024],
               hidden_dims_enc_common = [512,256],
               hidden_dims_enc_pre_Z = [256, 128],
               alpha = 3,
               narchetypes = 10,
               model_matrix = None, 
               fix_Z = False,
               Z_fix_norm = None,
               CUDA = True, 
               lr = 0.005,
               gamma_lr = 0.1,
               steps = 2000,
               inizialization_type = "best_ELBO_random", 
               torch_seed = 3, 
               batch_size = 5128):

    if CUDA and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')


    torch.manual_seed(torch_seed)
    pyro.clear_param_store()

    input_matrix = [torch.tensor(input_matrix_i) for input_matrix_i in input_matrix]
    if model_matrix is not None:
        model_matrix  =  torch.tensor(model_matrix)
    if normalization_factor is not None:
        normalization_factor = [torch.tensor(normalization_facotr_i) for normalization_facotr_i in normalization_factor]
    else:
         normalization_factor = [torch.ones(input.matrix[i].shape[0]) for i in range(len(input_matrix))]
    if side_matrices is not None:
        side_matrices =  [torch.tensor(side_matrices_i) for side_matrices_i in side_matrices]


    ### prepare parameters for deepAA

    n_dim_input = [inp.shape[1] for inp in input_matrix]
    if side_matrices is not None:
        n_dim_input_side = [inp.shape[1] for inp in side_matrices]
    else:
        n_dim_input_side = None



    deepAA = DeepAA(
               n_dim_input = n_dim_input, 
               input_types = input_types,
               n_dim_input_side = n_dim_input_side, 
               batch_size = batch_size, 
               input_types_side = input_types_side,
               hidden_dims_dec_common = hidden_dims_dec_common,
               hidden_dims_dec_last = hidden_dims_dec_last,
               hidden_dims_enc_ind = hidden_dims_enc_ind,
               hidden_dims_enc_common = hidden_dims_enc_common,
               hidden_dims_enc_pre_Z = hidden_dims_enc_pre_Z,
               alpha = alpha,
               narchetypes = narchetypes,
               fix_Z = fix_Z,
               Z_fix_norm = Z_fix_norm
               )
    lrd = gamma_lr ** (1 / steps)
    t = trange(steps, desc='Bar desc', leave = True)
    loss = Trace_ELBO
    svi = SVI(deepAA.model, deepAA.guide, optim.ClippedAdam({"lr": lr, "lrd" : lrd}), loss=loss())
    elbo_list = [] 

    for j in t:
        
        batch_indexes = sample_batch(input_matrix, model_matrix,batch_size)
        input_matrix_run = [input_matrix[i][batch_indexes,:] for i in range(len(input_matrix))]
        normalization_factor_run = [normalization_factor[i][batch_indexes] for i in range(len(normalization_factor))] 
        
        if model_matrix is not None:
            model_matrix_run = model_matrix[batch_indexes,:]
        else:
            model_matrix_run = None

        if  side_matrices is not None:
            side_matrices_run = [side_matrices[i][batch_indexes,:] for i in range(len(side_matrices))]
        else:
            side_matrices_run = None
        
        elbo = svi.step(input_matrix_run, model_matrix_run, 
                        normalization_factor_run, side_matrices_run, loss_weights_reconstruction, 
                        loss_weights_side)
        elbo_list.append(elbo )
        _ = t.set_description('ELBO: {:.5f}  '.format(elbo))
        _ = t.refresh()

    params_run = {}
    
    A,B,Z = deepAA.encoder(input_matrix)
    
    params_run["A"] = A
    params_run["B"] = B
    
    if fix_Z:
        params_run["Z"] = A @ Z
        archetypes_inferred = B @ A @ Z
        params_run["archetypes_inferred_fixed"] = Z
    else:
        params_run["Z"] = Z
        archetypes_inferred = B @ Z 

    params_run["archetypes_inferred"] = archetypes_inferred
    
    if model_matrix is not None:
        
        beta_mean = pyro.param("beta_means_param") 
        beta_cov = pyro.param("beta_cov_param") 
        
        params_run["beta_mean"] = beta_mean
        params_run["beta_cov"] = beta_cov
        
        eta_regr = torch.matmul(model_matrix, beta_mean.t())
        
        input_reconstructed, side_reconstructed = deepAA.decoder(A @ archetypes_inferred + eta_regr)
    else:
        input_reconstructed, side_reconstructed = deepAA.decoder(A @ archetypes_inferred)
        
    

    input_loss, side_loss,weights_reconstruction, weights_side, input_loss_no_reg,side_loss_no_reg, total_loss = deepAA.model(input_matrix_run, model_matrix_run, normalization_factor_run, side_matrices_run, loss_weights_reconstruction, loss_weights_side)
    
    params_run["input_loss"] = input_loss
    params_run["total_loss"] = total_loss
    params_run["input_loss_unreg"] = input_loss_no_reg
    params_run["total_loss"] = total_loss
    params_run["weights_input"] = weights_reconstruction
    if side_matrices is not None:
        params_run["side_reconstructed"] = side_reconstructed
        params_run["side_loss"] = side_loss
        params_run["weights_side"] = weights_side
        params_run["side_loss_unreg"] = side_loss_no_reg
        

    
    params_input = {
        
       "input_types" : input_types,
       "input_types_side" : input_types_side,
       "hidden_dims_dec_common" : hidden_dims_dec_common,
       "hidden_dims_dec_last" : hidden_dims_dec_last,
       "hidden_dims_enc_ind" : hidden_dims_enc_ind,
       "hidden_dims_enc_common" : hidden_dims_enc_common,
       "hidden_dims_enc_pre_Z" : hidden_dims_enc_pre_Z,
       "alpha" : alpha,
       "narchetypes" : narchetypes,
       "CUDA" : CUDA, 
       "lr" : lr,
       "steps" : steps,
       "inizialization_type" : inizialization_type, 
       "torch_seed" : torch_seed, 
       "batch_size" : batch_size

    }
    
    
    if CUDA and torch.cuda.is_available():
        params_run = {k:to_cpu_ot_iterate(v) for k,v in params_run.items() }
        
    params_run = {k:detach_or_iterate(v) for k,v in params_run.items()}
    
    params_run["highest_archetype"] = ["arc" + str(ar + 1) for ar in np.argmax(params_run["A"], axis = 1)]

    
    final_dict = {
        "inferred_quantities" : params_run,
        "hyperparametes" : params_input,
        "ELBO" : elbo_list,
    }
    
    return final_dict
    
