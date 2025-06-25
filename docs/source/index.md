# ggml-ot - Improve Optimal Transport applications by Global Ground Metric Learning on Distribution Classes

The Python package for "Global Ground Metric Learning with Applications to scRNA data" published at AISTATS2025. Learning a Global Metric as the Ground Metric of a Optimal Transport (OT) distance improves downstream-applications, such as classification, clustering and embeddings on synthetic and realworld scRNA-seq data. By learning a low-rank Mahalanobis distance between elements of distributions, ggml-ot learns interpretable low-dimensional subspaces that capture the class relations between distributions under OT. For scRNA-seq data containing patients at different disease stages, this corresponds to subspaces of the gene expression space realted to disease-related biological processes.

## Installation
To get started, install ggml-ot via pip:
```bash
pip install ggml-ot
```

## Citation

```bibtex
@misc{kühn2025globalgroundmetriclearning,
  title={Global Ground Metric Learning with Applications to scRNA data},
  author={Damin Kühn and Michael T. Schaub},
  year={2025},
  eprint={2506.15383},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.15383},
}
```

```{toctree}
---
maxdepth: 2
caption: Tutorials
---
tutorials
```


```{toctree}
---
maxdepth: 2
caption: API
---
api
```
