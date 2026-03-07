# Effeciency Experiments

These files are to try to do a fair comparison between pruning, knowledge distillation, and dendritic optimization.  The goal is to find a model and dataset where the following is true:

- There are at least to sizes of the architecture
- The smaller size is worse than the larger size on the dataset
- The model is not so overparameterized pruning doesnt matter, i.e. pruning the model negatively impacts scores

Once this is found, the experiment to perform is to add dendrites to the smaller model and compare large pruned vs small dendritic across parameter counts.
