# GGML-OT

<img src="https://github.com/DaminK/ggml-ot/blob/main/docs/source/images/icon_ggrouml.png?raw=True" width="300" />

## Abstract

Optimal transport (OT) provides a robust framework for comparing probability distributions.
Its effectiveness is significantly influenced by the choice of the underlying ground metric.
Traditionally, the ground metric has either been (i) predefined, e.g. as a Euclidean metric, or (ii) learned in a supervised way, by utilizing labeled data to learn a suitable ground metric for enhanced task-specific performance.
While predefined metrics often do not account for the inherent structure and varying significance of different features in the data, existing supervised ground metric learning methods often fail to generalize across multiple classes or are limited to distributions with shared supports.
To address this issue, this paper introduces a novel approach for learning metrics for arbitrary distributions over a shared metric space.
Our method differentiates elements like a global metric, but requires only class labels on a distribution-level for training akin a ground metric.
The resulting learned global ground metric enables more accurate OT distances, which can significantly improve clustering and classification tasks. It can create task-specific shared embeddings across elements of different distributions including unseen data.

## Installation

### Via pip

```terminal
pip install ggml-ot
```

### Manual

```terminal
git clone https://github.com/DaminK/ggml-ot
cd ggml-ot
pip install poetry
poetry lock && poetry install
```

### Development installation

```terminal
git clone https://github.com/DaminK/ggml-ot
cd ggml-ot
pip install poetry
peotry lock && poetry install --with dev
pre-commit install
```

## Citation

If you use this code in your research, please cite the following paper:
> Global Ground Metric Learning with Applications to scRNA data
>
> Damin Kuehn and Michael T. Schaub, Department of Computer Science RWTH Aachen
>
> Published at AISTATS2025 (DOI will follow)
