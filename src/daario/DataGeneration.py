import torch
import torch.distributions

def generate_synthetic_data(model, archetype_distribution, 
                            deterministic : bool = False, 
                            ncells : int = 1000 ,
                            dirichlet_variance_factor : float = 1.,
                            to_cpu = False,
                            seed = 3):
    """
    Generates synthetic data using a given model and archetype distribution parameters.

    Parameters:
    - model (dict): A dictionary containing the trained model and its parameters, including 'inferred_quantities' with 'archetypes_inferred' and the 'deepAA_obj'.
    - archetype_distribution (Tensor): A tensor representing the distribution of archetypes to be used for generating synthetic data.
    - deterministic (bool, optional): If True, generates data deterministically based on the mean of the distribution. Defaults to False.
    - ncells (int, optional): Number of synthetic data points (cells) to generate. Defaults to 1000.
    - dirichlet_variance_factor (float, optional): A scaling factor applied to the archetype distribution to adjust the variance of the Dirichlet distribution from which the synthetic archetype coefficients are sampled. Defaults to 1.0.
    - to_cpu (bool, optional): If True, moves generated tensors to CPU. Useful if the model is on a GPU and you want to analyze the data on CPU. Defaults to False.
    - seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 3.

    Returns:
    tuple: A tuple containing two elements:
      - generate_output (list of Tensors): The generated synthetic data.
      - side_output (list of Tensors or None): The generated side information, if available in the model; otherwise, None.

    The function first samples synthetic archetype coefficients from a Dirichlet distribution, then uses these coefficients to generate synthetic latent representations. These latent representations are then passed through the decoder of the provided model to generate synthetic data and, optionally, side information. The function allows for the generated data to be moved to CPU for further analysis.
    """
    
    torch.manual_seed(seed)
    A_synthetic = torch.distributions.Dirichlet(archetype_distribution * dirichlet_variance_factor).sample([ncells])
    if to_cpu:
        A_synthetic = A_synthetic.cpu()
    
    latent_synthetic = A_synthetic @ model["inferred_quantities"]["archetypes_inferred"]  
    
    
    if to_cpu:
        generate_output, side_output = model["deepAA_obj"].cpu().decoder(latent_synthetic)
        generate_output = [v.cpu() for v in generate_output]
        if side_output is not None:
            side_output = [v.cpu() for v in side_output]
    else:
        generate_output, side_output = model["deepAA_obj"].decoder(latent_synthetic)
        
    return generate_output, side_output


