# Cluster-PFN  

This repository contains the bare bones code for **Cluster-PFN**, a Transformer-based model that does Bayesian clustering.  

---

## Repository Structure  

- **`datasets/`**  
  Contains the real-world datasets used for testing missingness performance.  

- **`models/`**  
Stores  trained Cluster-PFN models. To see all models, go to the [anonymous Google Drive link](https://drive.google.com/drive/folders/17DDGVNKo6TF0Csp0DbFO5MuIYHfTSlli).


- **`notebooks/`**  
  Contains Jupyter notebooks for running experiments.  
  - **`experiments_missingness/`**: Experiments evaluating model performance under missingness.  
  - **`experiments_normal/`**: Experiments evaluating external clustering metrics.  

- **Python files**  
  - `transformer.py`: Model architecture.  
  - `prior.py`: GMM prior implementation.  
  - `utils.py`: Helper functions.  
  - `main.py`: Main training and evaluation entry point.  

---

## Notebooks  

### `experiments_normal/`  
- **`old_faithful.ipynb`**  
  Demonstrates Cluster-PFN performance on the Old Faithful dataset.  
- **`cluster_and_external_metrics.ipynb`**  
  Runs experiments for computing cluster accuracy and external metrics (ARI, AMI, purity, NLL).  
  - **Run**: Execute all cells.  
  - ⚠️ Full experiments are computationally expensive. For quicker runs, reduce the number of iterations (`iterations`) or the number of VI initializations (`n_inits`).  

### `experiments_missingness/`  
- **`missingness_vs_performance.ipynb`**  
  Evaluates performance vs. missingness on real-world datasets.  
  - **Run**: Execute all cells.  
  - For smaller experiments, reduce `iterations` or `n_inits`.  

---
