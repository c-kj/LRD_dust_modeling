# LRD_dust_modeling

[![DOI](https://zenodo.org/badge/992476241.svg)](https://doi.org/10.5281/zenodo.17380459)

Code for modeling dust re-emission and constraining dust extinction $A_V$ for Little Red Dots (LRDs).

This work has been published on ApJL.
Check out the paper for the description of methods and results: [Dust Budget Crisis in Little Red Dots (K. Chen et al. 2025)](https://doi.org/10.3847/2041-8213/ae1955).
You can also find the preprint on [arXiv:2505.22600](https://arxiv.org/abs/2505.22600).

The method is developed based on our previous work: [Little Red Dots: Rapidly Growing Black Holes Reddened by Extended Dusty Flows (Z. Li et al. 2025)](https://dx.doi.org/10.3847/1538-4357/ada5fb)

The analysis performed in our paper can be found in the Jupyter Notebook `notebooks/A_V_constraint.ipynb`.

## Citation

If you use this code in your research, please cite our paper [Dust Budget Crisis in Little Red Dots (K. Chen et al. 2025)](https://doi.org/10.3847/2041-8213/ae1955).

## Contact

If you are interested in trying the code, feel free to contact us through the email provided in the paper. For bug reports, please open an issue in this repository.

## Installation

First, download or clone this repository.

The recommended way to build up the environment is to use [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

This will create a virtual environment `.venv` in the project directory, with all the dependencies installed as specified in `pyproject.toml`.
Then, you can pick the python interpreter in `.venv` for your IDE or Jupyter Notebook.

In terminal, you can run scripts with `uv run your_script.py`, or activate the virtual environment manually:

```bash
source .venv/bin/activate
```
