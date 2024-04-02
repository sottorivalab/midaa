import torch
import re

from torch.functional import F

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def create_z_fix(dim_latent_space):
    """
    Adapted from https://github.com/bmda-unibas/DeepArchetypeAnalysis/blob/master/AT_lib/lib_at.py
    
    """
    z_fixed_t = torch.zeros([dim_latent_space, dim_latent_space + 1])
    
    for k in range(0, dim_latent_space):
        s = torch.tensor(0.0)
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2
            
        z_fixed_t[k, k] = torch.sqrt(1.0 - s)
        
        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]
                
            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = z_fixed_t.t()
    return z_fixed

def sample_batch(input_matrix, model_matrix, batch_size):
    
    return sample_batch_aux(input_matrix[0].shape[0], batch_size)
  

def sample_batch_aux(max_value, batch_size):
    ret = torch.randperm(torch.tensor(max_value).item())
    rest = max_value % batch_size
    N_splits = max_value // batch_size
    ret_list = [ret[(i * batch_size):((i+1) * batch_size)] for i in range(N_splits)]
    if rest > 0:
        ret_list.append(ret[-rest:])
    return ret_list


def sample_batch_by_class_aux(max,model_matrix ,batch_size):
    max_sample = torch.max(max, torch.tensor(batch_size))
    ret = torch.randperm(max_sample.item, size = (batch_size, ))
    return ret

def to_cpu_ot_iterate(x):
    if isinstance(x, list):
        ret = [val.cpu() for val in x]
    else:
        ret = x.cpu()
        
    return ret
    

def detach_or_iterate(x):
    if isinstance(x, list):
        ret = [val.detach().numpy() for val in x]
    else:
        ret = x.detach().numpy()
        
    return ret
    


def find_matching_indexes(input_list, regex_pattern):
    # Compile the regular expression for better performance if used repeatedly
    pattern = re.compile(regex_pattern)
    
    # Use list comprehension to find indexes
    matching_indexes = [index for index, element in enumerate(input_list) if pattern.search(element)]
    
    return matching_indexes

def select_elements_not_at_indexes(elements, exclude_indexes):
    # Use list comprehension to filter out elements at specified indexes
    return [element for index, element in enumerate(elements) if index not in exclude_indexes]

def select_elements_at_indexes(elements, include_indexes):
    # Use list comprehension to filter elements at specified indexes
    return [element for index, element in enumerate(elements) if index in include_indexes]



    
    
    