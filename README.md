# ARS-N Numerical Experiment Framework

This repository is a small research framework for optimization experiments centered on the ARS-N family:

- `ARS-RN`: Anchored Randomized Subspace Regularized Newton
- `ARS-CN`: Anchored Randomized Subspace Cubic Newton

The codebase is designed for controlled comparisons against related large-scale second-order methods such as:

- `RS-RN`
- `RS-CN`
- `Newton-CG`
- `Full Newton`

The system is YAML-driven and supports the standard workflow:

1. generate synthetic data
2. run optimization
3. save CSV logs and metadata
4. plot selected curves from the logs

## Purpose

The main goal of this project is to run reproducible numerical experiments for ARS-N style methods and their baselines.

In particular, the framework is intended to support:

- experiments on synthetic quadratic problems
- experiments on synthetic logistic regression problems
- comparison of anchored and non-anchored randomized subspace methods
- logging of optimization traces for later plotting and analysis

## Directory Structure

```text
.
├── data/
│   └── generated/
├── input/
│   ├── generate_data/
│   ├── optimize/
│   └── plot/
├── output/
│   ├── meta/
│   ├── plots/
│   └── results/
├── src/
│   ├── algorithms/
│   ├── plotting/
│   ├── problems/
│   ├── utils/
│   ├── cli.py
│   ├── config.py
│   └── registry.py
└── tests/
```

### Folder descriptions

`data/generated/`  
Stores generated datasets in `.npz` format. For example, synthetic quadratic data and synthetic logistic regression data are saved here before optimization runs.

`input/generate_data/`  
Contains YAML files for dataset generation.

`input/optimize/`  
Contains YAML files for optimization experiments. Each file specifies the problem, initialization, optimizer configuration, logging, and metadata output paths.

`input/plot/`  
Contains YAML files for plotting CSV histories.

`output/results/`  
Stores CSV log files produced by optimization runs.

`output/meta/`  
Stores JSON metadata and resolved YAML configurations for each run. This directory is also used for local matplotlib cache files during plot generation.

`output/plots/`  
Stores generated figures.

`src/algorithms/`  
Contains optimizer implementations. The ARS-N family lives under `src/algorithms/ars_n/`.

`src/problems/`  
Contains problem definitions and synthetic data generators.

`src/plotting/`  
Contains CSV-based plotting utilities.

`src/utils/`  
Contains shared utilities such as logging, timers, sketch operators, path helpers, YAML/JSON I/O, and seeding.

`tests/`  
Reserved for smoke tests and regression tests.

## Core Design

All optimizers work with a minimal problem interface:

```python
f(x) -> float
grad(x) -> np.ndarray
hvp(x, v) -> np.ndarray
```

The framework intentionally uses Hessian-vector products instead of dense Hessian formation for the large-scale methods.

Initial points are not part of the problem object. They are specified in optimize YAML files.

## Implemented Problems

### Quadratic

`src/problems/quadratic.py` implements synthetic diagonal quadratic problems with:

- closed-form objective
- closed-form gradient
- closed-form Hessian-vector product
- synthetic spectrum generation
- `.npz` save/load support

### Logistic Regression

`src/problems/logistic.py` implements L2-regularized logistic regression with:

- synthetic data generation
- dense and CSR input matrix support
- closed-form objective, gradient, and Hessian-vector product
- `.npz` save/load support

The current experiments use synthetic logistic data saved in `data/generated/`.

## Implemented Methods

### ARS-N family

The ARS-N family is implemented under `src/algorithms/ars_n/`.

`ARS-RN`  
Anchored randomized subspace Newton with:

- RK-based anchor construction
- anchored basis `Q_k = [q_0, S_k^T]`
- projected HVP-based reduced Hessian
- diagonal-shift regularization
- Armijo backtracking

Implementation:

- `src/algorithms/ars_n/ars_rn.py`
- `src/algorithms/ars_n/rk.py`

`ARS-CN`  
Anchored randomized subspace cubic Newton with:

- the same RK anchor mechanism as ARS-RN
- the same anchored basis construction
- reduced-space cubic regularization
- RS-CN style accept/reject logic based on `rho`

Implementation:

- `src/algorithms/ars_n/ars_cn.py`
- `src/algorithms/ars_n/rk.py`

Note:

- `src/algorithms/ars_n/main.py` is still a stub for a generic `ars_n` entry point
- the concrete runnable anchored methods are `ars_rn` and `ars_cn`

### Comparison methods

`RS-RN`  
Randomized subspace Newton without anchoring.

Implementation:

- `src/algorithms/rs_rn/main.py`

`RS-CN`  
Randomized subspace cubic Newton without anchoring.

Implementation:

- `src/algorithms/rs_cn/main.py`

`Newton-CG`  
Large-scale HVP-based baseline using conjugate gradient.

Implementation:

- `src/algorithms/newton_cg/main.py`

`Full Newton`  
Dense baseline for small problems. This method explicitly reconstructs the Hessian from repeated HVP calls and is intended only for verification or small-scale comparisons.

Implementation:

- `src/algorithms/full_newton/main.py`

### Registered but not yet implemented

The following names are still registered as placeholders:

- `ars_n`
- `rn`
- `cn`

These require additional mathematical specifications before they can be used as real solvers.

## Configuration Files

The workflow is completely driven by YAML files.

### Generate-data configs

Examples:

- `input/generate_data/quadratic_exp_001.yml`
- `input/generate_data/logistic_dense_001.yml`
- `input/generate_data/logistic_sparse_beta_001.yml`

### Optimize configs

Examples:

- `input/optimize/ars_rn_logistic_dense_001.yml`
- `input/optimize/ars_cn_logistic_dense_001.yml`
- `input/optimize/rs_rn_quadratic_exp_001.yml`
- `input/optimize/rs_cn_quadratic_exp_001.yml`
- `input/optimize/rs_cn_logistic_dense_001.yml`
- `input/optimize/newton_cg_quadratic_exp_001.yml`
- `input/optimize/full_newton_quadratic_exp_001.yml`

### Plot configs

Example:

- `input/plot/plot.yml`

## Command Examples

### 1. Generate data

Generate synthetic quadratic data:

```bash
python -m src.cli generate --config input/generate_data/quadratic_exp_001.yml
```

Generate synthetic logistic regression data:

```bash
python -m src.cli generate --config input/generate_data/logistic_dense_001.yml
```

### 2. Run optimization

Run ARS-RN on logistic regression:

```bash
python -m src.cli optimize --config input/optimize/ars_rn_logistic_dense_001.yml
```

Run ARS-CN on logistic regression:

```bash
python -m src.cli optimize --config input/optimize/ars_cn_logistic_dense_001.yml
```

Run RS-CN on logistic regression:

```bash
python -m src.cli optimize --config input/optimize/rs_cn_logistic_dense_001.yml
```

Run a quadratic baseline:

```bash
python -m src.cli optimize --config input/optimize/newton_cg_quadratic_exp_001.yml
```

### 3. Plot results

Create a plot from CSV logs:

```bash
python -m src.cli plot --config input/plot/plot.yml
```

## Logging

When `log.enabled: true`, each optimization run writes a CSV file to `output/results/`.

Every log row contains the required base columns:

1. `iter`
2. `f`
3. `grad_norm`
4. `step_norm`
5. `step_size`
6. `cumulative_time`
7. `per_iter_time`

These columns are shared across all methods so downstream analysis remains consistent.

After the base columns, each optimizer may append method-specific fields.

### Typical ARS-RN log fields

ARS-RN logs may additionally contain fields such as:

- `f_prev`, `f_next`
- `grad_norm_prev`, `grad_norm_next`
- `actual_reduction`
- `accepted`
- `armijo_iters`
- `gtd`
- `eta`
- `lambda_min_phpt`, `lambda_max_phpt`
- `lambda_shift`
- `cond_phpt_reg`
- `projected_grad_norm`
- `u_norm`
- `y_norm`
- `rk_residual_norm_init`, `rk_residual_norm_final`
- `hvp_calls_iter`, `hvp_calls_cum`
- `rk_hvp_calls_iter`, `rk_hvp_calls_cum`
- `subprob_hvp_calls_iter`, `subprob_hvp_calls_cum`

### Typical ARS-CN log fields

ARS-CN logs may additionally contain fields such as:

- `f_prev`, `f_next`
- `grad_norm_prev`, `grad_norm_next`
- `actual_reduction`
- `accepted`
- `gtd`
- `rho`
- `sigma`
- `model_decrease`
- `actual_decrease`
- `lambda_value`
- `cubic_solver`
- `projected_grad_norm`
- `u_norm`
- `y_norm`
- `rk_residual_norm_init`, `rk_residual_norm_final`
- `hvp_calls_iter`, `hvp_calls_cum`
- `rk_hvp_calls_iter`, `rk_hvp_calls_cum`
- `subprob_hvp_calls_iter`, `subprob_hvp_calls_cum`

### Metadata files

When `save_meta.enabled: true`, the framework also writes:

- a JSON metadata file to `output/meta/`
- a resolved YAML configuration to `output/meta/`

The metadata JSON includes:

- run name
- seed
- algorithm name
- problem name
- final objective value
- final gradient norm
- iteration count
- termination status
- optional git commit hash
- path to the CSV history

### Logger flushing

`RunLogger` writes CSV files incrementally and flushes to disk periodically.
The flush frequency can be controlled with:

```yaml
log:
  enabled: true
  csv_path: output/results/example.csv
  flush_every: 50
```

If `flush_every` is omitted, the default is `50`.

## Plotting

The plotting system reads CSV files by column name.

You can choose arbitrary logged columns for the x-axis and y-axis, for example:

- `iter`
- `cumulative_time`
- `f`
- `grad_norm`

The plotting config specifies:

- input CSV files
- labels
- x column
- y column
- scales
- title and axis labels
- output figure path

## Progress Output In The Terminal

When `optimizer.verbose: true`, the optimizer prints:

- one header line at the beginning of the run
- one short progress line every `print_every` iterations

The progress lines are intentionally compact and show:

- iteration index
- objective value
- gradient norm

## Sketch Operators

Randomized subspace methods use Gaussian sketches.

The current sketch configuration supports:

```yaml
sketch:
  mode: operator
  block_size: 256
  dtype: float64
```

The RK subroutine used by ARS-RN and ARS-CN can be configured in the same spirit:

```yaml
rk:
  T: 20
  r: 20
  seed_offset: 0
  mode: operator
  block_size: 256
  dtype: float64
```

Internally, the framework now materializes each Gaussian sketch once per use and reuses it within the iteration to avoid repeated random-number generation overhead.

## Setup

Example environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

Typical dependencies:

- `numpy`
- `PyYAML`
- `matplotlib`
- `pytest`
- `scipy` for sparse logistic datasets
- `torchvision` only if the optional MNIST helper is used

## Notes

- The framework is intentionally lightweight and avoids heavy abstractions.
- Large-scale methods are expected to use only `f`, `grad`, and `hvp`.
- `Full Newton` is not a scalable method and should only be used on small problems.
- The repository is set up for experiment scripting, not for packaging a polished optimization library.
