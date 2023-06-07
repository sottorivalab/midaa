import numpy as np

def get_input_params_adata(adata, is_normalized = True):
    if is_normalized:
        input_data = [adata.X]
        normalization = [np.ones(adata.X.shape[0])]
        input_distribution = ["G"]
    else:
        input_data = [adata.X]
        normalization = [adata.obs["n_counts"] / np.sum(adata.obs["n_counts"])]
        input_distribution = ["NB"]
    return input_data, normalization, input_distribution


def add_to_obs_adata(inf_res, adata):
    col_names = [ "arc" + str(i) for i in np.arange(res["hyperparametes"]["narchetypes"])]
    for i in range(len(col_names)):
        adata.obs[col_names[i]] = inf_res["inferred_quantities"]["A"][:,i]
    adata.obsm["X_aa"] = inf_res["inferred_quantities"]["Z"]
    return adata


