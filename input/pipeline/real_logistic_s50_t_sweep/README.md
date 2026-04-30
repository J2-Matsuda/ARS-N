# Real Logistic s=50 / T Sweep Pipelines

This experiment set solves each real logistic dataset with:

- `ARS-CN`: `subspace_dim: 50`, `rk.r: 50`, `rk.T: 50` and `100`, `max_iter: 10000`
- `RS-CN`: `subspace_dim: 50`, `max_iter: 10000`
- `RS-RN`: `subspace_dim: 50`, `max_iter: 10000`
- `GD`: `max_iter: 100000`
- `AGD-Unknown`: `max_iter: 100000`

| dataset | optimize configs |
|---|---|
| `epsilon_normalized` | `ars_cn_t50`, `ars_cn_t100`, `rs_cn`, `rs_rn`, `gd`, `agd_unknown` |
| `real_sim` | `ars_cn_t50`, `ars_cn_t100`, `rs_cn`, `rs_rn`, `gd`, `agd_unknown` |
| `rcv1_binary` | `ars_cn_t50`, `ars_cn_t100`, `rs_cn`, `rs_rn`, `gd`, `agd_unknown` |
| `gisette` | `ars_cn_t50`, `ars_cn_t100`, `rs_cn`, `rs_rn`, `gd`, `agd_unknown` |

Run all datasets:

```bash
python -m src.cli pipeline --config input/pipeline/real_logistic_s50_t_sweep.yml
```

Run one dataset:

```bash
python -m src.cli pipeline --config input/pipeline/real_logistic_s50_t_sweep/gisette.yml
```
