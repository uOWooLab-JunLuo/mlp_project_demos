# Generating Graphs from MOF Structures

This is a quick demo of how to generate a graph from a MOF structure (a ".cif" file).

## How it works

Please read through the fully documented [**DEMO**](demo.ipynb) Jupyter Notebook in this repo (either online via Github or locally when downloaded).

## Batch generation

A [**SCRIPT**](batch_generation.py) for generating the graphs in parallel (utlizing pyhton's multiprocessing) is included in this repo. Please modify `n_procs`, `src_path`, `dst_path` before use. Exceptions may occured and is not dealt with in this script; if raised, just skip the generation for that structure.

## Scores to beat

To date, our best models are trained using the [CatBoost](https://catboost.ai/) library (a variant of gradient boosted decision trees). A [PREVIEW](https://github.com/uOWooLab-JunLuo/milar-preview) of the models are available here. The following table shows the results.

| Element (Group) | R2 score | MAE | RMSE |
| :----: | :----: | :----: | :----: |
| C | 0.049 | 0.077 | 0.95 |
| H | 0.017 | 0.029 | 0.88 |
| O | 0.024 | 0.043 | 0.91 |
| N | 0.036 | 0.065 | 0.97 |
| Zn | 0.020 | 0.061 | 0.86 |
| Cu | 0.025 | 0.053 | 0.88 |
| F | 0.013 | 0.023 | 0.80 |
| Cl | 0.018 | 0.030 | 0.87 |
| Br | 0.021 | 0.038 | 0.77 |
| S | 0.027 | 0.052 | 0.97 |
| Fe | 0.015 | 0.038 | 0.93 |
| V | 0.041 | 0.059 | 0.77 |
| I | 0.026 | 0.039 | 0.95 |
| Co | 0.036 | 0.059 | 0.87 |
| Others | 0.062 | 0.123 | 0.93 |

Elements not listed above are included in Others*.
