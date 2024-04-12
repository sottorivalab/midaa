def rank_features_by_arch(X, inference_result, var_names, scale = True,  plot = True):
    
    arch_matrix = inference_result["inferred_quantities"]["B"] @ X
    
    if scale:
        means = np.mean(arch_matrix, axis=0)
        stds = np.std(arch_matrix, axis=0)
        arch_matrix = (arch_matrix - means) / stds
    
    # Calculating the absolute difference of each gene's expression from the mean expression across all samples
    absolute_differences_matrix = np.abs(arch_matrix - np.mean(arch_matrix, axis=0))
 
    # Identifying the top 5 genes for each sample based on absolute differences
    top_5_genes_per_sample_indices = np.argsort(absolute_differences_matrix, axis=1)[:, -5:]

    # Creating gene IDs for the top 5 genes for each sample
    top_5_genes_per_sample_ids = np.array([[var_names[i] for i in row] for row in top_5_genes_per_sample_indices])

    top_5_genes_per_sample_ids

    