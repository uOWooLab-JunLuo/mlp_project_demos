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
| C | 0.8703 | 0.0808 | 0.1266 |
| H | 0.6938 | 0.0287 | 0.0559 |
| O | 0.7712 | 0.0426 | 0.0864 |
| N | 0.9149 | 0.0595 | 0.1007 |
| Zn | 0.9465 | 0.0329 | 0.0671 |
| Cu | 0.9251 | 0.0242 | 0.0525 |
| S | 0.9110 | 0.0350 | 0.0620 |
| Fe | 0.8206 | 0.0495 | 0.0684 |
| Halogen* | 0.8359 | 0.0277 | 0.0523 |
| Others** | 0.8930 | 0.1136 | 0.2317 |

Halogens* includes the F, Cl, Br, I elements. Elements not listed above are included in Others**.
