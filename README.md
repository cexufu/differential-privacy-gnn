# Differential privacy for GNN-based recommendation

Part of my UCL MSc dissertation (Knowledge, Information and Data Science), which compared privacy-preserving techniques for graph neural network recommender systems: differential privacy, federated learning and homomorphic encryption. This repository holds the differential-privacy experiments.

**Governance question:** how much recommendation quality, compute and memory does a platform give up when it perturbs user-preference signals to protect privacy, and how does that trade-off change with graph size and sparsity?

## What the notebook does

`Differential_Privacy.ipynb` contains the full pipeline:

1. **Data.** A small book-recommendation graph of users, books, ratings and follower relations (121 nodes, 386 edges).
2. **Model.** A three-layer GCN (PyTorch Geometric) that scores user-book pairs, combining graph structure with user-preference embeddings and rule-based scores.
3. **Privacy mechanism.** Gaussian noise of a configurable scale is added to the preference embeddings and rule scores before they enter the prediction.
4. **Utility and cost sweep.** Training is repeated five times at each noise scale, logging loss, runtime and memory use.
5. **Scalability and sparsity tests.** Synthetic graphs that preserve the original node ratios and edge ratios are generated, with Laplace-noise anonymisation of attributes, to see how cost and loss change with graph size and sparsity.

## Limitations

- The noise injection is a perturbation mechanism in the spirit of differential privacy. The notebook does not compute a formal (ε, δ) privacy guarantee for the trained model.
- The dataset is small, so the results show trends rather than production-scale numbers.
- Utility is measured by training loss only, not by ranking metrics such as recall@k or NDCG.

## Requirements

Python 3, PyTorch, PyTorch Geometric, pandas, scikit-learn, psutil, matplotlib, Faker.
