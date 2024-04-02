import torch
import torch
import pyro
import numpy as np
import pyro.optim as optim
import random
from pyro.infer.autoguide import AutoDelta, AutoNormal, AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, TraceEnum_ELBO,JitTrace_ELBO
from tqdm import trange


from scdeepaa.Utils import sample_batch, to_cpu_ot_iterate, detach_or_iterate, find_matching_indexes, select_elements_not_at_indexes, select_elements_at_indexes
from scdeepaa.deepAA import DeepAA


def fit_DAARIO(input_matrix,
                normalization_factor = None,
                input_types = ["NB"],
                loss_weights_reconstruction = None,
                side_matrices = None, 
                input_types_side = None,
                loss_weights_side = None,
                hidden_dims_dec_common = [256,512],
                hidden_dims_dec_last = [1024],
                hidden_dims_dec_last_side = None,
                hidden_dims_enc_ind = [1024],
                hidden_dims_enc_common = [512,256],
                hidden_dims_enc_pre_Z = [256, 128],
                layers_independent_types = None,
                layers_independent_types_side = None,
                image_size = [256,256],
                alpha = 3,
                narchetypes = 10,
                model_matrix = None, 
                just_VAE = False,
                linearize_encoder = False,
                linearize_decoder = False,
                VAE_steps = None,
                CUDA = True, 
                lr = 0.005,
                gamma_lr = 0.1,
                steps = 2000,
                fix_Z = False,
                initialization_B_weight = None,
                Z_fix_norm = None,
                Z_fix_release_step = None,
                reconstruct_input_and_side = False,
                initialization_input = None,
                initialization_steps_phase_1 = 1000,
                initialization_lr_phase_1 = 0.001,
                initialization_steps_phase_2 = 350,
                initialization_lr_phase_2 = 0.0005,
                torch_seed = 3, 
                batch_size = 5128,
                kernel_size=3, 
                stride=1, 
                padding=1, 
                pool_size=2, 
                pool_stride=2):

    """
    Fits the DAARIO model to given input data, using specified architecture and optimization parameters.

    Parameters:
    - input_matrix (list of ndarray): Input data matrix, where each entry is a tensor representation
      of the input data for a different modality.
    - normalization_factor (list of ndarray, optional): Normalization factors for the input data. Default is None.
    - input_types (list of str, optional): Types of input data. Default is ["NB"].
    - loss_weights_reconstruction (list of float, optional): Weights for the reconstruction loss. Default is None.
    - side_matrices (list of ndarray, optional): Side information matrices. Default is None.
    - input_types_side (list of str, optional): Types of side information. Default is None.
    - loss_weights_side (list of float, optional): Weights for the side information loss. Default is None.
    - hidden_dims_dec_common, hidden_dims_dec_last, hidden_dims_dec_last_side,
      hidden_dims_enc_ind, hidden_dims_enc_common, hidden_dims_enc_pre_Z (list of int, optional): 
      Dimensions of various layers in the decoder and encoder. Defaults are specified for each.
    - layers_independent_types, layers_independent_types_side (list of str, optional): Types of layers for independent modeling.
    - image_size (list of int, optional): Size of the input images (width, height). Default is [256, 256].
    - alpha (float, optional): Hyperparameter alpha. Default is 3.
    - narchetypes (int, optional): Number of archetypes to model. Default is 10.
    - model_matrix (ndarray, optional): Matrix representing the model. Default is None.
    - just_VAE (bool, optional): Flag to run only the VAE without the archetypal analysis. Default is False.
    - linearize_encoder, linearize_decoder (bool, optional): Flags to linearize encoder and decoder. Defaults are False.
    - VAE_steps (int, optional): Number of steps to run the VAE. Default is None.
    - CUDA (bool, optional): Flag to use CUDA if available. Default is True.
    - lr (float, optional): Learning rate for the optimizer. Default is 0.005.
    - gamma_lr (float, optional): Learning rate decay factor. Default is 0.1.
    - steps (int, optional): Number of training steps. Default is 2000.
    - fix_Z (bool, optional): Flag to fix Z during training. Default is False.
    - initialization_B_weight (float, optional): Initial weight for B. Default is None.
    - Z_fix_norm (float, optional): Normalization factor for Z when fixed. Default is None.
    - Z_fix_release_step (int, optional): Step to release Z fix. Default is None.
    - reconstruct_input_and_side (bool, optional): Flag to reconstruct both input and side information. Default is False.
    - initialization_input (dict, optional): Initial values for the input. Default is None.
    - initialization_steps_phase_1, initialization_steps_phase_2 (int, optional): 
      Number of steps for the two phases of initialization. Defaults are specified.
    - initialization_lr_phase_1, initialization_lr_phase_2 (float, optional): 
      Learning rates for the two phases of initialization. Defaults are specified.
    - torch_seed (int, optional): Seed for PyTorch's RNG. Default is 3.
    - batch_size (int, optional): Batch size for training. Default is 5128.
    - kernel_size, stride, padding, pool_size, pool_stride (int, optional): 
      Convolution and pooling parameters. Defaults are specified.

    Returns:
    dict: A dictionary containing the final parameters, input parameters, ELBO list, and the DeepAA instance.

    This function configures and trains a DeepAA model based on the specified parameters and data.
    It handles device setting, seed initialization, data preprocessing, and the training loop, including
    potential initialization phases for the model. The final output includes training diagnostics and the
    trained model itself, ready for further analysis or application.
    """
    
    using_CUDA = False
    if CUDA and torch.cuda.is_available():
        using_CUDA = True
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')


    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    random.seed(torch_seed)
    
    pyro.clear_param_store()

    input_matrix = [torch.tensor(input_matrix_i).float() for input_matrix_i in input_matrix]
    if model_matrix is not None:
        model_matrix  =  torch.tensor(model_matrix).float()
    if normalization_factor is not None:
        normalization_factor = [torch.tensor(normalization_facotr_i).float() for normalization_facotr_i in normalization_factor]
    else:
         normalization_factor = [torch.ones(input_matrix[i].shape[0]) for i in range(len(input_matrix))]
    if side_matrices is not None:
        side_matrices =  [torch.tensor(side_matrices_i).float() for side_matrices_i in side_matrices]
    
    if initialization_input is not None:
        initialization_input = {k:torch.tensor(v).float() for k,v in initialization_input.items()}

    if Z_fix_release_step is None:
        Z_fix_release_step = steps - round(steps * 0.3)

    

    ### prepare parameters for deepAA

    n_dim_input = [inp.shape[1] for inp in input_matrix]
    if side_matrices is not None:
        n_dim_input_side = [inp.shape[1] for inp in side_matrices]
    else:
        n_dim_input_side = None

    initialization_mode = False
    
    if initialization_input is not None:
        initialization_mode = True
    

    deepAA = DeepAA(
                n_dim_input = n_dim_input, 
                input_types = input_types,
                n_dim_input_side = n_dim_input_side, 
                batch_size = batch_size, 
                input_types_side = input_types_side,
                hidden_dims_dec_common = hidden_dims_dec_common,
                hidden_dims_dec_last = hidden_dims_dec_last,
                hidden_dims_dec_last_side = hidden_dims_dec_last_side,
                hidden_dims_enc_ind = hidden_dims_enc_ind,
                hidden_dims_enc_common = hidden_dims_enc_common,
                hidden_dims_enc_pre_Z = hidden_dims_enc_pre_Z,
                alpha = alpha,
                narchetypes = narchetypes,
                fix_Z = fix_Z,
                Z_fix_norm = Z_fix_norm,
                #initialization_mode = initialization_mode,
                just_VAE = just_VAE,
                linearize_encoder = linearize_encoder,
                linearize_decoder = linearize_decoder,
                layers_independent_types = layers_independent_types,
                layers_independent_types_side = layers_independent_types_side,
                image_size = image_size,
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                pool_size=pool_size, 
                pool_stride=pool_stride
               )
    
    if initialization_mode:
        print("Running initialization!")
        
        deepAA, elbo_list_init_1  = training_loop(deepAA = deepAA, 
                                input_matrix = input_matrix,
                                model_matrix = model_matrix, 
                                normalization_factor = normalization_factor, 
                                side_matrices = side_matrices, 
                                initialization_input = initialization_input, 
                                loss_weights_reconstruction = loss_weights_reconstruction,  
                                loss_weights_side = loss_weights_side, 
                                initialization_B_weight = initialization_B_weight, 
                                batch_size = batch_size, lr = initialization_lr_phase_1, 
                                
                                steps = initialization_steps_phase_1, lrd = 1, 
                                VAE_steps = VAE_steps, fix_Z = fix_Z, 
                               Z_fix_release_step = Z_fix_release_step, CUDA = using_CUDA,
                                                 initialization_mode_step1 = True)

        deepAA, elbo_list_init_2  = training_loop(deepAA = deepAA, 
                                input_matrix = input_matrix,
                                model_matrix = model_matrix, 
                                normalization_factor = normalization_factor, 
                                side_matrices = side_matrices, 
                                initialization_input = initialization_input, 
                                loss_weights_reconstruction = loss_weights_reconstruction,  
                                loss_weights_side = loss_weights_side, 
                                initialization_B_weight = initialization_B_weight, 
                                batch_size = batch_size, lr = initialization_lr_phase_2, 
                                steps = initialization_steps_phase_2, lrd = 1, 
                                VAE_steps = VAE_steps, fix_Z = fix_Z, 
                               Z_fix_release_step = Z_fix_release_step, CUDA = using_CUDA,
                                                 initialization_mode_step2 = True)
        

    if VAE_steps is None:
        VAE_steps = 0
        just_VAE = True
    
    if steps > 0:
        lrd = gamma_lr ** (1 / steps)
    else:
        lrd = 1
    
    deepAA, elbo_list  = training_loop(deepAA = deepAA, 
                                        input_matrix = input_matrix,
                                        model_matrix = model_matrix, 
                                        normalization_factor = normalization_factor, 
                                        side_matrices = side_matrices, 
                                        initialization_input = initialization_input, 
                                        loss_weights_reconstruction = loss_weights_reconstruction,  
                                        loss_weights_side = loss_weights_side, 
                                        initialization_B_weight = initialization_B_weight, 
                                        batch_size = batch_size, lr = lr, 
                                        steps = steps, lrd = lrd,
                                        VAE_steps = VAE_steps, fix_Z = fix_Z, 
                                       Z_fix_release_step = Z_fix_release_step, CUDA = using_CUDA)
    
    params_run = {}
    
    A,B,Z = deepAA.encoder(input_matrix)
    
    params_run = initialize_params(params_run,A,B)
    
    params_run = handle_z(params_run, fix_Z, Z_fix_release_step, steps, Z, A, B)
    
    params_run = handle_model_matrix(params_run, model_matrix, reconstruct_input_and_side,
                                     A, deepAA)
    
    params_run =  calculate_loss(params_run, fix_Z, deepAA, input_matrix, model_matrix, normalization_factor, 
                                 side_matrices, loss_weights_reconstruction, loss_weights_side)
    


    
    params_input = {
        "normalization_factor": normalization_factor,
        "input_types": input_types,
        "loss_weights_reconstruction": loss_weights_reconstruction,
        "input_types_side": input_types_side,
        "loss_weights_side": loss_weights_side,
        "hidden_dims_dec_common": hidden_dims_dec_common,
        "hidden_dims_dec_last": hidden_dims_dec_last,
        "hidden_dims_dec_last_side": hidden_dims_dec_last_side,
        "hidden_dims_enc_ind": hidden_dims_enc_ind,
        "hidden_dims_enc_common": hidden_dims_enc_common,
        "hidden_dims_enc_pre_Z": hidden_dims_enc_pre_Z,
        "layers_independent_types": layers_independent_types,
        "layers_independent_types_side": layers_independent_types_side,
        "image_size": image_size,
        "alpha": alpha,
        "narchetypes": narchetypes,
        "model_matrix": model_matrix,
        "just_VAE": just_VAE,
        "linearize_encoder": linearize_encoder,
        "linearize_decoder": linearize_decoder,
        "VAE_steps": VAE_steps,
        "CUDA": CUDA,
        "lr": lr,
        "gamma_lr": gamma_lr,
        "steps": steps,
        "fix_Z": fix_Z,
        "initialization_B_weight": initialization_B_weight,
        "Z_fix_norm": Z_fix_norm,
        "Z_fix_release_step": Z_fix_release_step,
        "reconstruct_input_and_side": reconstruct_input_and_side,
        "initialization_steps_phase_1": initialization_steps_phase_1,
        "initialization_lr_phase_1": initialization_lr_phase_1,
        "initialization_steps_phase_2": initialization_steps_phase_2,
        "initialization_lr_phase_2": initialization_lr_phase_2,
        "torch_seed": torch_seed,
        "batch_size": batch_size,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "pool_size": pool_size,
        "pool_stride": pool_stride
    }
    
    params_run = prepare_final_parameters(params_run, CUDA)
    


    
    final_dict = compile_final_dict(params_run, params_input, elbo_list, deepAA)
    
    return final_dict
    

def training_loop(deepAA, input_matrix, model_matrix, 
                  normalization_factor, side_matrices, 
                     initialization_input, loss_weights_reconstruction, 
                  loss_weights_side, initialization_B_weight, 
                  batch_size, lr, steps, lrd, 
                  VAE_steps , fix_Z, Z_fix_release_step,
                 initialization_mode_step1 = False, initialization_mode_step2 = False,
                 CUDA = True):
    
    if steps == 0:
        return deepAA, []
    
    t = trange(steps, desc='Bar desc', leave=True)
    
    if  initialization_mode_step1:
        # Exiting initialization mode
        deepAA.initialization_mode_step1 = True
    if  initialization_mode_step2:
        # Exiting initialization mode
        deepAA.initialization_mode_step2 = True
    
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    
    
    
    
    elbo_list_init = []

    for j in t:
        elbo_epoch = []
        batch_indexes = sample_batch(input_matrix, model_matrix, batch_size)
        if not initialization_mode_step1 and not initialization_mode_step2:
            if fix_Z and j >= Z_fix_release_step:
                deepAA.Z_fix_release = True

            if j > VAE_steps and VAE_steps > 0:
                deepAA.just_VAE = False

        
        for tt in range(len(batch_indexes)):
            
            input_matrix_run = [input_matrix[i][batch_indexes[tt],:] for i in range(len(input_matrix))]
            normalization_factor_run = [normalization_factor[i][batch_indexes[tt]] for i in range(len(normalization_factor))] 

            if model_matrix is not None:
                model_matrix_run = model_matrix[batch_indexes[tt],:]
            else:
                model_matrix_run = None

            if  side_matrices is not None:
                side_matrices_run = [side_matrices[i][batch_indexes[tt],:] for i in range(len(side_matrices))]
            else:
                side_matrices_run = None

            if  initialization_mode_step1 or initialization_mode_step2:
                initialization_input_run = { "B" : initialization_input["B"][:,batch_indexes[tt]] }
            else:
                initialization_input_run = None

            if j == 0 and tt == 0:
                with pyro.poutine.trace(param_only=True) as param_capture:
                    _ = loss_fn(deepAA.model, deepAA.guide, input_matrix_run, model_matrix_run, 
                              normalization_factor_run, side_matrices_run, loss_weights_reconstruction, 
                              loss_weights_side, initialization_input_run, initialization_B_weight)

                params = [site["value"].unconstrained()
                    for site in param_capture.trace.nodes.values()]
                names = [site
                    for site in param_capture.trace.nodes.keys()]
                regex_B = "encoder\$\$\$layers_latent\.1\.[0-9]*"

                

                if initialization_mode_step1:
                    params = set(select_elements_not_at_indexes(params, find_matching_indexes(names, regex_B)))
                elif initialization_mode_step2:
                    params_old = params
                    params = set(select_elements_at_indexes(params, find_matching_indexes(names, regex_B)))
                else:
                    params = set(params)
                optimizer = pyro.optim.clipped_adam.ClippedAdam(params, lr=lr, betas=(0.90, 0.999), lrd = lrd)
            
            #print(list(params)[1])
            #print(list(params_old)[1])

                
            elbo = loss_fn(deepAA.model, deepAA.guide, input_matrix_run, model_matrix_run, 
                              normalization_factor_run, side_matrices_run, loss_weights_reconstruction, 
                              loss_weights_side, initialization_input_run, initialization_B_weight)
            elbo.backward()
            # take a step and zero the parameter gradients
            optimizer.step()
            optimizer.zero_grad()
            elbo_epoch.append(elbo)

        # Calculate and log the mean ELBO for the current epoch
        mean_elbo = sum(elbo_epoch) / len(batch_indexes)
        if CUDA:
            elbo_list_init.append(mean_elbo.cpu().detach().numpy())
        else:
            elbo_list_init.append(mean_elbo.detach().numpy())
        _ = t.set_description('ELBO: {:.5f}  '.format(mean_elbo))
        _ = t.refresh()
    
    if  initialization_mode_step1:
        # Exiting initialization mode
        deepAA.initialization_mode_step1 = False
    if  initialization_mode_step2:
        # Exiting initialization mode
        deepAA.initialization_mode_step2 = False

    return deepAA, elbo_list_init



def initialize_params(params_run, A, B):
    params_run = {"A": A, "B": B}
    return params_run

def handle_z(params_run, fix_Z, Z_fix_release_step, steps, Z, A, B):
    if fix_Z:
        if Z_fix_release_step < steps:
            params_run["archetypes_inferred_fixed"] = pyro.param("archetypes_params")
            params_run["archetypes_Z_fixed"] = Z
        else:
            params_run["archetypes_inferred_fixed"] = Z
        params_run["Z"] = A @ params_run["archetypes_inferred_fixed"]
        archetypes_inferred = B @ A @ params_run["archetypes_inferred_fixed"]
    else:
        params_run["Z"] = Z
        archetypes_inferred = B @ Z
    params_run["archetypes_inferred"] = archetypes_inferred
    return params_run

def handle_model_matrix(params_run, model_matrix, reconstruct_input_and_side, A, deepAA):
    
    archetypes_inferred = params_run["archetypes_inferred"] 
    if model_matrix is not None:
        beta_mean = pyro.param("beta_means_param") 
        beta_cov = pyro.param("beta_cov_param") 
        params_run["beta_mean"] = beta_mean
        params_run["beta_cov"] = beta_cov
        if reconstruct_input_and_side:
            eta_regr = torch.matmul(model_matrix, beta_mean.t())
            input_reconstructed, side_reconstructed = deepAA.decoder(A @ archetypes_inferred + eta_regr)
            params_run["input_reconstructed"] = input_reconstructed
            params_run["side_reconstructed"] = side_reconstructed
    else:
        if reconstruct_input_and_side:
            input_reconstructed, side_reconstructed = deepAA.decoder(A @ archetypes_inferred)
            params_run["input_reconstructed"] = input_reconstructed
            params_run["side_reconstructed"] = side_reconstructed
    return params_run

def calculate_loss(params_run, fix_Z, deepAA, input_matrix, model_matrix, normalization_factor, side_matrices, loss_weights_reconstruction, loss_weights_side):
    if fix_Z:
        input_loss, side_loss, weights_reconstruction, weights_side, input_loss_no_reg, side_loss_no_reg, total_loss, Z_loss_no_reg, Z_loss = deepAA.model(input_matrix, 
            model_matrix, normalization_factor, side_matrices, loss_weights_reconstruction, loss_weights_side)
        params_run["Z_loss"] = Z_loss
        params_run["Z_loss_unreg"] = Z_loss_no_reg
    else:
        input_loss, side_loss, weights_reconstruction, weights_side, input_loss_no_reg, side_loss_no_reg, total_loss = deepAA.model(input_matrix, 
            model_matrix, normalization_factor, side_matrices, loss_weights_reconstruction, loss_weights_side)
    
    params_run["input_loss"] = input_loss
    params_run["total_loss"] = total_loss
    params_run["input_loss_unreg"] = input_loss_no_reg
    params_run["weights_input"] = weights_reconstruction
    if side_matrices is not None:
        params_run["side_loss"] = side_loss
        params_run["weights_side"] = weights_side
        params_run["side_loss_unreg"] = side_loss_no_reg
    return params_run

def prepare_final_parameters(params_run, CUDA):
    if CUDA and torch.cuda.is_available():
        params_run = {k:to_cpu_ot_iterate(v) for k,v in params_run.items()}
    params_run = {k:detach_or_iterate(v) for k,v in params_run.items()}
    params_run["highest_archetype"] = ["arc" + str(ar + 1) for ar in np.argmax(params_run["A"], axis=1)]
    return params_run

def compile_final_dict(params_run, params_input, elbo_list, deepAA):
    final_dict = {
        "inferred_quantities": params_run,
        "hyperparameters": params_input,
        "ELBO": elbo_list,
        "deepAA_obj": deepAA
    }
    return final_dict

def load_model_from_state_dict(model,input_matrix, path, CUDA = False):
    
    """
    Loads a model's state dictionary from a specified path and updates the model with inferred quantities
    based on a provided input matrix.

    Parameters:
    - model (dict): A dictionary containing the model structure, including the 'deepAA_obj' (the model object)
      and 'hyperparameters'. This dictionary will be updated with 'inferred_quantities' based on the input data.
    - input_matrix (list of ndarray): The input data matrix to be used for inference after loading the model,
      where each entry is a tensor representation of the input data for a different modality.
    - path (str): Path to the file containing the saved state dictionary of the model.
    - CUDA (bool, optional): Indicates whether CUDA (GPU) should be used for loading and inference. If False,
      operations will be performed on the CPU. Default is False.

    The function loads the model's state dictionary from the specified path, considering whether CUDA is enabled
    or not. It sets the model to evaluation mode, processes the provided input matrix, and performs inference to
    obtain and update the model with new inferred quantities such as A, B, and Z matrices. It also updates the
    model with any final parameters adjustments based on the model's hyperparameters.

    Note:
    - The 'model' dictionary must contain 'deepAA_obj', an instance of the model, and 'hyperparameters', a dictionary
      specifying model configurations like 'fix_Z', 'Z_fix_release_step', and 'steps'.
    - After loading the state and performing inference, the function updates the 'model' dictionary with
      'inferred_quantities', which include the results from the inference.
    """
    
    if CUDA:
        model["deepAA_obj"].load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    else:
        model["deepAA_obj"].load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
    model["deepAA_obj"].eval()
    
    input_matrix = [torch.tensor(input_matrix_i).float() for input_matrix_i in input_matrix]
    A,B,Z = model["deepAA_obj"].encoder(input_matrix)
    
    params_run = {}
    
    params_run = initialize_params(params_run,A,B)
    params_run = handle_z(params_run, model["hyperparameters"]["fix_Z"], model["hyperparameters"]["Z_fix_release_step"],
                          model["hyperparameters"]["steps"], Z, A, B)
    
    params_run = prepare_final_parameters(params_run,  model["hyperparameters"]["CUDA"])
    model["inferred_quantities"] = params_run