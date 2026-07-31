# ABTeaLab: Bayesian A/B Testing for Estimating Heterogeneous Treatment Effects in Online Advertising

Code accompanying the paper **"Estimating Heterogeneous Treatment Effects in Online Advertising Experiments: A Hierarchical Bayesian Approach"** by Aljoša Vodopija, Anže Alič, Tim Poštuvan, Martin Jakomin, and Blaž Škrlj (Teads), published at the AdKDD 2026 workshop at the ACM SIGKDD Conference on Knowledge Discovery and Data Mining.

## Abstract

We present a production hierarchical Bayesian framework for estimating heterogeneous treatment effects in large-scale online advertising experiments. By jointly modeling campaign-level effects and cross-campaign heterogeneity, it addresses a limitation of inverse-variance weighting, the standard meta-analytic baseline, which becomes unstable under high heterogeneity and sparse events. Across simulations spanning a range of effect sizes and heterogeneity regimes, the approach delivers more reliable inference under high heterogeneity while matching or improving estimation accuracy in most settings. We further demonstrate its practical value in a production demand-side platform that handles over 5 million requests per second, integrating it with a sequential testing procedure and evaluating it across three experiments covering over 35,000 campaigns from more than 3,000 advertisers.

## Model

![Plate diagram of the hierarchical Bayesian model](https://raw.githubusercontent.com/outbrain-inc/ab-tea-lab/master/assets/model_diagram.png)

Plate diagram of the proposed hierarchical Bayesian model. Unshaded circles are latent variables, shaded circles are observed, the diamond is a deterministic node, and bare symbols are fixed constants.

## ab-tea-lab

`ab-tea-lab` is the Python library behind the paper: a meta-analysis toolkit for CPA-focused A/B tests. It ships a data simulator for generating synthetic campaign-level experiments, several models for estimating treatment effects (from the classic inverse-variance baseline to the hierarchical Bayesian model above), and an evaluator for comparing them on metrics such as bias, coverage, and RMSE.

## Installation

Install from source:

```bash
git clone https://github.com/outbrain-inc/ab-tea-lab.git
cd ab-tea-lab
pip install .
```

## Dev setup

1. From the package's root directory (the one containing `pyproject.toml` and `src/`, following the [src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)), create and activate a virtual environment.
2. Use the targets in `Makefile` to work on the package:
    - `make install` — install the package with its test and lint dependencies
    - `make test` — run the test suite
    - `make format` — auto-format the code
    - `make lint` — check for lint issues

## Usage

### Running a simulation experiment

An `Experiment` ties together a data-generating simulator, one or more models, and an evaluator, then runs the models against many simulated datasets to compare their performance:

```python
from ab_tea_lab.simulator.random import RandomSimulator
from ab_tea_lab.model.inverse_variance import InverseVariance
from ab_tea_lab.model.bayesian_poisson_regression import BayesianPoissonRegression
from ab_tea_lab.evaluator.evaluator import Evaluator
from ab_tea_lab.experiment import Experiment

# 1. Configure a data-generating process
simulator = RandomSimulator(
    num_advertisers=50,
    campaigns_mean=5.0,
    spend_mean=470.0,
    spend_sd=1175.0,
    cpa_mean=20.0,
    cpa_sd=10.0,
    cpa_within_sd=5.0,
    effect_mean=0.0,   # 0 = no treatment effect (null scenario)
    effect_sd=0.1,
    equal_spend=True,
    random_seed=42,
)

# 2. Pick one or more models to compare
models = [
    InverseVariance(method_re="iterated"),
    BayesianPoissonRegression(),
]

# 3. Run the experiment
experiment = Experiment(
    simulator=simulator,
    models=models,
    evaluator=Evaluator(),
    n_simulations=10,
)
results = experiment.run()

for name, evaluation in results.items():
    print(f"{name}: bias={evaluation.bias:.4f}, coverage={evaluation.coverage:.2f}, rmse={evaluation.rmse:.4f}")
```

### Inspecting simulated data

Each call to a simulator's `run()` method returns a single simulated dataset, which can be inspected directly:

```python
data = simulator.run()
data.data.head()            # pandas DataFrame with ctrl/treat spend & conversions
data.plot_diagnostics()     # distributions, scatter plots, and summary table
```

### Available models

| Model | Description |
|---|---|
| `InverseVariance` | Fixed- or random-effects inverse-variance meta-analysis (frequentist) |
| `BayesianPoissonRegression` | Hierarchical Poisson regression via PyMC (Bayesian) |

## Citing this work
If you use or build on top of ab-tea-lab, feel free to cite:

```
@inproceedings{vodopija2026estimating,
author = {Vodopija, Aljo\v{s}a and Ali\v{c}, An\v{z}e and Po\v{s}tuvan, Tim and Jakomin, Martin and \v{S}krlj, Bla\v{z}},
title = {Estimating Heterogeneous Treatment Effects in Online Advertising Experiments: A Hierarchical Bayesian Approach},
year = {2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the AdKDD 2026 Workshop at the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
numpages = {6},
location = {Jeju, Korea},
series = {KDD '26}
}
```