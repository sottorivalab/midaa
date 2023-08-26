# multiDeepAA

multideepaa is a package designed for performing Deep Archetypal Analysis on multiomics data. This package is designed to help researchers, bioinformaticians, and data scientists uncover the hidden archetypes in complex, high-dimensional multiomics datasets.

## Installation

```bash
$ pip install multideepaa
```

## Quick

multiDeepAA can be run on a single datset in anndata(scanpy) format by just running 

```python
import multideepaa as aa
import scanpy as sc

adata =  sc.datasets.pbmc3k_processed()
input_matrix, norm_factors, input_distribution = aa.get_input_params_adata(adata)

narchetypes = 5

aa_result = scdeepaa.fit_deepAA(
    input_matrix,
    norm_factors,
    input_distribution,
    narchetypes = narchetypes)


```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`multideepaa` was created by Salvatore Milite. It is licensed under the terms of the MIT license.

## Credits

`multideepaa` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
