# Deep Archetypal Analysis for Representation and Learning of Omics data (DAARIO)

DARRIO is a package designed for performing Deep Archetypal Analysis on multiomics data. The documentation can be find here [https://sottorivalab.github.io/daario/](https://sottorivalab.github.io/daario/)

<img src="https://github.com/sottorivalab/daario/logo.png" width="400px" align="left">

*The package is under active development, expect breaking changes and incomplete documentat for a bit*
*I'll try my best to speed this up, if something is broken or you need help please open an issue, dontt be shy!*


## Installation

```{bash}
# Soon on pypi
git clone https://github.com/sottorivalab/daario.git
# you need poetry installed
poetry install 
```

## Quick Start


DAARIO encodes you multi-modal data into a latent simplex: 

$$
\mathbf{Z^*} =   \mathbf{A}  \mathbf{B}  \mathbf{Z} 
$$


DAARIO leans the matrices $\mathbf{A}$, $\mathbf{B}$ and $\mathbf{Z}$ in an amortized fashion, namely we learn a function that takes in input the different data modalities $\mathbf{X_g}$ indexed by $g` and learns the 3 matrices. As you could have got from the name, we parametrize the function as a neural network. 
The network is implemented in a Variational Autoencdoer fashion, so we have and encoding and decoding function as well as probabilistic definition of the matrix factorization problem above.
Both the encoder and the decoder have a shared portion where data fusion occurs and an independent piece where modality specific encoding and decoding takes place.


If you are happy with that we have some cool tutorials that will show you how to use DAARIO on real [single modality](https://sottorivalab.github.io/daario/scRNA_single_modality.html) and [multimodal](https://sottorivalab.github.io/daario/scMulti_multimodal.ipynb) data.

Otherwise the best way to start is to read [this](https://sottorivalab.github.io/daario/daario_long_form.html) or the companion [paper](https://www.biorxiv.org/content/10.1101/2024.04.05.588238v1) and understand what DAARIO actually does in details and what are the [parameters](https://sottorivalab.github.io/daario/implementation_and_parameters.ipynb) you can play with.


A minimal example to run the tool:

```{python}

import multideepaa as daa
import scanpy as sc

adata =  sc.datasets.pbmc3k_processed()
input_matrix, norm_factors, input_distribution = daa.get_input_params_adata(adata)

narchetypes = 5

aa_result = daa.fit_deepAA(
    input_matrix,
    norm_factors,
    input_distribution,
    narchetypes = narchetypes
    )


```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## ToDOs  (slow but steady):  🔨

- [ ] Final API Documnetation
- [ ] Tutorial math on AA
- [ ] Tutorial parameters and networks
- [ ] Quick start single modality
- [ ] Quick start multiomics
- [ ] Allow the user to specify its own encoder/decoder
- [ ] Provide some module builders
- [ ] Test batch/covariate correction in latent space 

## Citation 

If you have used DAARIO in your research, consider citing:
```{bibtex}
@article {milite2024,
	author = {Salvatore Milite and Giulio Caravagna and Andrea Sottoriva},
	title = {Interpretable Multi-Omics Data Integration with Deep Archetypal Analysis},
	year = {2024},
	doi = {10.1101/2024.04.05.588238},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/09/2024.04.05.588238},
	journal = {bioRxiv}
}
```

## License

`daario` was created by Salvatore Milite. It is licensed under the terms of the MIT license.

## Credits

`daario` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).