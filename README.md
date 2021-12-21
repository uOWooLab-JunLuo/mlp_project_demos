# A quick demo of generating graphs from MOF structures

This is a quick demo of how to generate a graph from a MOF structure (a ".cif" file).

## How to use

Please read through the fully documented `mof_graph.ipynb` Jupyter Notebook in this repo (either online via Github or locally when downloaded). Copy code snippets and write your own code that suits your needs. Clone this repo if necessary.

## Scores to beat

With our hand-crafted descriptors/features and fully connected neural networks (FCNNs), these are the lastest results of our test sets. Note that we categorized our data into elements or element groups to train our FCNNs (i.e. each element has its own FCNNs).

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
