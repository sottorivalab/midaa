# Deep Archetypal Analysis for Representation and Learning of Omics data (DAARIO)

DARRIO is a package designed for performing Deep Archetypal Analysis on multiomics data. This package is designed to help researchers, bioinformaticians, and data scientists uncover the hidden archetypes in complex, high-dimensional multiomics datasets.

## Installation

```bash
$ pip install daario
```

## Quick Start


DAARIO encodes you multi-modal data into a latent simplex: 

$$ \mathbf{Z^*} =   \mathbf{A}  \mathbf{B}  \mathbf{Z} $$

DAARIO leans the matrices $\mathbf{A}$, $\mathbf{B}$ and $\mathbf{Z}$ in an amortized fashion, namely we learn a function that takes in input the different data modalities $\mathbf{X_g}$ indexed by $g$ and learns the 3 matrices. As you could have got from the name, we parametrize the function as a neural network. The network is implemented in a Variaitonal Autoencdoer fashion, so we have and encoding and decoding function as well as probabilistic definition of the matrix factorization problem above.
Both the encoder and the decoder have a shared portion where data fusion occurs and an independent piece where modality specific encoding and decoding takes place.


If you are happy with that we have some cool tutorials that will show you how to use DAARIO on real data.

Otherwise the best way to start is to understand what DAARIO actually does in details and what are the parameters you can play with.


A minimal example to run the tool:

```python

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

## ToDOs  (slow but steady) :chart_with_upwards_trend:

- [ ] Allow the user to specify its own encoder/decoder
- [ ] Provide some module builders
- [ ] Test batch/covariate correction in latent space 
- [ ] Example notebook CLL MOFA
- [ ] Example notebook CD34+  

## License

`daario` was created by Salvatore Milite. It is licensed under the terms of the MIT license.

## Credits

`daario` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).