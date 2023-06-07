import scipy
from python_tsp.exact import solve_tsp_dynamic_programming
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np




def plot_archetypes_simplex(res, distance_type = "euclidean", cmap = "nipy_spectral", color_by = None):

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
    aa = res["inferred_quantities"]["A"]
    labels_rad = np.radians(labels)
    r = (np.sqrt( (aa * np.cos(labels_rad)).sum(axis=1)**2 + (aa * np.sin(labels_rad)).sum(axis=1)**2))
    theta = np.arctan2((aa * np.sin(labels_rad)).sum(axis=1) , (aa * np.cos(labels_rad)).sum(axis=1))
    arc_names = [ "arc" + str(i+1) for i in permutation]

    if color_by is None:
        colors = theta
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        scatter = ax.scatter(theta, r, c=colors, cmap=cmap, alpha=r)
        ax.set_thetagrids(labels, arc_names)
        ax.set_rmax(1.2)
        ax.set_rticks([0.5, 1])  # Less radial ticks
        ax.set_rlabel_position(-2.5)  # Move radial labels away from plotted line
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        legend1 = ax.legend(labels = colors_dict.keys(), handles = scatter.legend_elements()[0],
                        loc="right", title="Classes")
        #ax.add_artist(legend1)
        ax.grid(True)

        ax.set_title("Archetype projection in a 2d polytope", va='bottom')
        plt.show()
    else:
        colors_dict = { k:mpl.colors.to_hex(v) for k,v in zip(color_by.unique(), mpl.colormaps[cmap].colors[:len(color_by.unique())])}
        #colors = color_by.map(colors_dict).values.tolist()
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        for col in color_by.unique():
            idx = color_by == col
            scatter = ax.scatter(theta[idx], r[idx], c=colors_dict[col], alpha=r[idx], label = col)
            
        ax.set_thetagrids(labels, arc_names)
        ax.set_rmax(1.2)
        ax.set_rticks([0.5, 1])  # Less radial ticks
        ax.set_rlabel_position(-2.5)  # Move radial labels away from plotted line
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        legend = ax.legend(bbox_to_anchor=(1.5, 1), loc="upper left", title="Groups")
        for lh in legend.legendHandles:
            lh.set_alpha(1.0)
        ax.grid(True)

def plot_ELBO(res):
    losses = res["ELBO"]
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
