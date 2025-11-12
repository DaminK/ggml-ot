```{toctree}
:hidden: true
:maxdepth: 1

installation
tutorials/index
api/index
contributors_guide
references
```


[![PyPI](https://img.shields.io/pypi/v/ggml-ot)](https://pypi.org/project/ggml-ot/)
[![PyPI Downloads](https://img.shields.io/pepy/dt/ggml-ot?logo=pypi)](https://pepy.tech/project/ggml-ot)
[![Docs](https://app.readthedocs.org/projects/ggml-ot/badge/?version=latest)](https://ggml-ot.readthedocs.io)
[![CI](https://github.com/DaminK/ggml-ot/actions/workflows/pytest.yml/badge.svg)](https://github.com/DaminK/ggml-ot/actions/workflows/pytest.yml)


# ggml-ot: Supervised Optimal Transport

Learning the Ground Metrics of **O**ptimal **T**ransport (**OT**) improves downstream applications, such as classification, clustering, trajectory inference and embeddings. **G**lobal **G**round **M**etric **L**earning (**GGML**) learns low-dimensional subspaces that capture class relations between distributions under OT.

The focus of this package is to improve OT analyses on single-cell data in the [Anndata](https://anndata.readthedocs.io/en/latest/index.html) format, allowing exchangeability with other tools from [Scanpy](https://scanpy.readthedocs.io/en/stable/) and [Scverse](https://scverse.org/packages/). It also supports generic data as [numpy.ndarray](https://numpy.org/doc/stable/index.html).

## Improve OT on Single-cell data {material-round}`biotech;1em;sd-text-info`

A common analysis on single-cell data is investigating genetic differences between patient groups (e.g. disease states). One approach is to capture these differences as OT distances, where each patient/sample is considered as a distribution of cells over the measured expressed/regulated genes. However, computing OT with common Ground Metrics, such as the Euclidean or Cosine distance, often fails to capture the relations between patient groups due to noise, biological fluctuations and disease-unrelated genetic differences, as shown below:

<style>
table th:first-of-type {
    width: 33%;
}
table th:nth-of-type(2) {
    width: 33%;
}
table th:nth-of-type(3) {
    width: 33%;
}
</style>

**<p style="text-align: center;"> Optimal Transport between Patients [^1] with: </p>**
| GGML |  Euclidean Ground Metric | Cosine Ground Metric
|:-----:|:-----:|:-----:|
|![](images/myocard_example/umap_ggml.png) | ![](images/myocard_example/umap_euclidean.png) | ![](images/myocard_example/umap_cosine.png)


GGML uses the patient groups as distribution labels to learn a suitable ground metric that captures the patient groups under OT. It is weakly supervised and does not require any cell type annotation or filtering, allowing to investigate group-related biological processes (e.g. disease mechanism) in the learned gene subspaces, across known and unknown cell subtypes.

This method was first introduced in [*Global Ground Metric Learning with Applications to scRNA data*](https://proceedings.mlr.press/v258/kuhn25a.html) published at AISTATS2025.


## Installation {octicon}`plug;1em;sd-text-info`
The easiest way to install ggml-ot is from PyPI via pip:
```bash
pip install ggml-ot
```

## Getting Started on AnnData {octicon}`rocket;1em;sd-text-info`
In this small example, we demonstrate how to train ggml-ot to perform Supervised Optimal Transport on AnnData. To use Cross-Validation, Hyperparameter Tuning and more, refer to [the tutorials](/tutorials/index).
```python
  import ggml_ot

  # Load Anndata from CELLxGENE
  id = "c1f6034b-7973-45e1-85e7-16933d0550bc.h5ad"
  adata = ggml_ot.data.load_cellxgene(id)

  # Setup Dataset from Anndata
  dataset = ggml_ot.from_anndata(adata, patient_col="sample", label_col="patient_group")

  # Train GGML on all patients
  dataset.train()
```

```{figure} images/get_started/clustermap_emb.png
:width: 700px
:align: center

Clustermap and Embedding of Patients with OT distances using the learned ground metric
```

The dataset contains the AnnData object in `.adata`. After training on the dataset, the AnnData object contains the learned gene subspace of the ground metric in `.varm["W_ggml"]` and the cells embedded in this subspace in `.obsm["X_ggml"]`.

```python
import scanpy as sc

# Access adata with learned metric
adata = dataset.adata

# Rank genes of learned components (adata.varm["W_ggml"])
ggml_ot.gene.ranking(adata,gene_symbols="feature_name")

# Show cells embedded in low-dimensional gene subspace (adata.obsm["X_ggml"])
sc.pl.embedding(adata,basis="X_ggml",color=["patient_group",'CDH19','STAB1'],
                gene_symbols="feature_name",use_raw=False,legend_loc='on data')
```
```{figure} images/get_started/gene_ranking_ggml.png
:width: 750px
:align: center

Ranking of loadings in learned gene subspace
```

```{figure} images/get_started/cell_emb_ggml.png
:width: 800px
:align: center

Embedding of cells in learned gene subspace with 2 example marker genes from ranking W_ggml1
```

## Citation {material-round}`format_quote;1em;sd-text-info`

If you are using this package for you research and find it helpful, please use this reference:

```text
Kühn, Damin, and Michael T. Schaub. "Global Ground Metric Learning with Applications to scRNA data." Proceedings of the 28th International Conference on Artificial Intelligence and Statistics. PMLR, 2025.
```
*In BibTeX format:*
```bibtex
@misc{kuehn2025ggml,
  title={Global Ground Metric Learning with Applications to scRNA data},
  author={Kühn, Damin and Schaub, Michael T.},
  booktitle = 	 {Proceedings of the 28th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3295--3303},
  year = 	 {2025},
  volume = 	 {258},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {03--05 May},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v258/main/assets/kuhn25a/kuhn25a.pdf},
  url = 	 {https://proceedings.mlr.press/v258/kuhn25a.html},
}
```

[^1]: Patient-level scRNA-seq dataset from *Kuppe, Christoph, et al. "Spatial multi-omic map of human myocardial infarction." Nature (2022)*.
