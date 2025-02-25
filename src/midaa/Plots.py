import scipy
from python_tsp.exact import solve_tsp_dynamic_programming
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np




def plot_archetypes_simplex(res, distance_type = "euclidean", cmap = "nipy_spectral", color_by = None, subsample = None, s = None, l_size = 30, l_title = "Group"):
    
    """
    Plots the archetypes inferred by the model on a simplex represented in polar coordinates, optionally coloring
    points by a given attribute.

    Parameters:
    - res (dict): A dictionary containing model results, specifically 'inferred_quantities' with archetype coefficients 'A'.
    - distance_type (str, optional): The type of distance metric to use for determining the order of archetypes. Defaults to "euclidean".
    - cmap (str, optional): Colormap for plotting. Defaults to "nipy_spectral".
    - color_by (Series, optional): Pandas series or similar containing labels to color data points by. Defaults to None.
    - subsample (array-like, optional): Indices to subsample the archetype coefficients 'A' for plotting. Defaults to None.
    - s (float, optional): Size of points in the plot. Defaults to None.
    - l_size (int, optional): Size of labels in the legend. Defaults to 30.
    - l_title (str, optional): Title of the legend. Defaults to "Group".

    Returns:
    tuple: (fig, ax) where 'fig' is the figure object and 'ax' is the axes object of the plot.

    The function computes distances between archetypes to determine their order, maps these onto a circle,
    and plots them in polar coordinates. Points can be colored by a categorical variable if provided. The function
    aims to provide an intuitive visualization of the relationship between archetypes, highlighting their relative
    distances and potential clustering.
    """
    
    if subsample:
        arcs = res["inferred_quantities"]["A"][subsample,:]
    else:
        arcs = res["inferred_quantities"]["A"]
    dist_matrix = scipy.spatial.distance.cdist(arcs.T,arcs.T,distance_type)
    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    dists = []
    for i in range(1, len(permutation)):
        dists.append(dist_matrix[permutation[i-1], permutation[i]])

    dists.append(dist_matrix[permutation[-1], permutation[0]])
    dists = np.array(dists)  
    dists = dists /dists.sum()
    narchs = (res["inferred_quantities"]["A"]).shape[1]
    labels = 360 * dists
    labels = labels.cumsum()[:-1]
    labels = np.append(0,labels)
    aa = res["inferred_quantities"]["A"][:,permutation]
    labels_rad = np.radians(labels)
    r = (np.sqrt( (aa * np.cos(labels_rad)).sum(axis=1)**2 + (aa * np.sin(labels_rad)).sum(axis=1)**2))
    theta = np.arctan2((aa * np.sin(labels_rad)).sum(axis=1) , (aa * np.cos(labels_rad)).sum(axis=1))
    arc_names = [ "arc" + str(i+1) for i in permutation]

    if color_by is None:
        colors = theta
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        scatter = ax.scatter(theta, r, c=colors, cmap=cmap, alpha=r, s = s)
        ax.set_thetagrids(labels, arc_names)
        ax.set_rmax(1.2)
        ax.set_rticks([0.5, 1])  # Less radial ticks
        ax.set_rlabel_position(-2.5)  # Move radial labels away from plotted line
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.tick_params(axis='both', which='major', pad=15)
        #legend1 = ax.legend(labels = arc_names, handles = scatter.legend_elements()[0],
        #                loc="right", title="Classes")
        #ax.add_artist(legend1)
        
        ax.grid(True)

        ax.set_title("Archetype projection in a 2d polytope", va='bottom')
        #plt.show()
    else:
        colors_dict = { k:mpl.colors.to_hex(v) for k,v in zip(color_by.unique(), mpl.colormaps[cmap].colors[:len(color_by.unique())])}
        #colors = color_by.map(colors_dict).values.tolist()
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        for col in color_by.unique():
            idx = color_by == col
            scatter = ax.scatter(theta[idx], r[idx], c=colors_dict[col], alpha=r[idx], label = col, s = s)
        
        ax.set_thetagrids(labels, arc_names)
        ax.set_rmax(1.2)
        ax.set_rticks([0.5, 1])  # Less radial ticks
        ax.set_rlabel_position(-2.5)  # Move radial labels away from plotted line
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.tick_params(axis='both', which='major', pad=15)

        legend = ax.legend(bbox_to_anchor=(1.2, 1), loc="upper left", title=l_title)
        for lh in legend.legendHandles:
            lh._sizes = [l_size]
        fig.legend = legend
        ax.grid(True)
    
    return fig, ax

def plot_ELBO(res):
    """
    Plots the ELBO loss over SVI steps from the results of model fitting.

    Parameters:
    - res (dict): A dictionary containing the ELBO loss values in the key 'ELBO'.

    This function creates a line plot showing how the ELBO loss evolved during the optimization process. It's useful
    for assessing the convergence of the model fitting process. The x-axis represents the SVI step, and the y-axis
    represents the ELBO loss value at that step.
    """
    losses = res["ELBO"]
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    
    
    
def plot_ELBO_across_runs(res_dictionary, warmup = 500):    

    """
    Plots a comparison of ELBO losses across different model runs, post-warmup phase.

    Parameters:
    - res_dictionary (dict): A dictionary where keys are descriptive names of model runs (e.g., number of archetypes)
      and values are dictionaries containing the 'ELBO' loss values for those runs.
    - warmup (int, optional): Number of initial steps to exclude from the plot to focus on the post-warmup phase. Defaults to 500.

    The function creates a boxplot for each key in the res_dictionary, showing the distribution of ELBO values
    across steps after the specified warmup phase. This visualization is helpful for comparing the model fitting
    performance across different configurations or hyperparameter settings, especially to identify which setups
    lead to better convergence based on the ELBO loss metric.
    """
    ELBOS = {k:obj["ELBO"][warmup:] for k,obj in res_dictionary.items()}
    ELBO_plot = pd.DataFrame(ELBOS)
    ELBO_plot =pd.melt(ELBO_plot)
    ELBO_plot["value"] = ELBO_plot["value"].astype("float")
    plt.clf()
    sns.boxplot(data=ELBO_plot, x = "variable", y = "value")
    plt.xlabel("N archetypes")
    plt.ylabel("ELBO")
    plt.show()

    
def plot_archetypes_proportion(adata, by = "cluster"):
    pass
    
