# Molecule-Text Retrieval

Project for the ALTEGRAD (Advanced AI for Texts and Graphs) class in the Master Data Science of IP Paris. 
Realized by Antoine Gilson, Paul Lemoine Vandermoere and Alexandre Zenou. 

Graph neural network for molecular graph and text description retrieval.

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

The data comes from the kaggle challenge "Molecular Graph Captioning" Challenge on Kaggle (https://www.kaggle.com/competitions/molecular-graph-captioning/data)

Place your preprocessed graph data files in the `data/` directory:
- `train_graphs.pkl`
- `validation_graphs.pkl`
- `test_graphs.pkl`

## Usage

Run the following scripts in order:

### 1. Inspect Graph Data

Check the structure and contents of your graph files:

```bash
python inspect_graph_data.py
```

### 2. Generate Description Embeddings with a GTE model

Create embeddings for molecular descriptions:

```bash
python generate_description_embeddings.py
```

This generates:
- `data/train_embeddings.csv`
- `data/validation_embeddings.csv`

Main updates compared to the baseline:
- Support for multiple pretrained text encoders (including scientific and retrieval-optimized models)
- Increased maximum sequence length to reduce truncation
- Mean pooling over non-padding tokens instead of CLS-only pooling
- L2 normalization of embeddings for stable cosine similarity
- Robust handling of empty or malformed descriptions

### 3. Train à GINE-Conv Model

Train the graph neural network:

```bash
python train_gcn.py
```

Main updates compared to the baseline:
- Use of explicit node and edge features instead of feature-free graphs
- GINE-based message passing to incorporate edge attributes
- Multi-pooling readout (mean, max, sum) at the graph level
- Projection head with normalization and dropout
- Contrastive training objective (InfoNCE-style) instead of embedding regression
- L2-normalized graph embeddings for retrieval

This creates a model `model_{ARCH}_{POOL}_{JK_MODE}_output.pt`.

### 4. Run a top-k weighted Retrieval System

Retrieve descriptions for test molecules:

```bash
python retrieval_answer.py
```

Main updates compared to the baseline:
- Replacement of hard nearest-neighbor retrieval with a weighted top-k strategy
- Aggregation of multiple close textual candidates using similarity-based weights
- Improved robustness to noise in the embedding space

This generates `test_retrieved_descriptions.csv` with retrieved descriptions for each test molecule.

## Output

- `model_{ARCH}_{POOL}_{JK_MODE}_output.pt`: Trained GCN model
- `test_retrieved_descriptions.csv`: Retrieved descriptions for test set

### Performance

The proposed modifications consistently improved retrieval performance compared to the baseline pipeline, with gains primarily driven by stronger text embeddings and contrastive alignment between graph and text representations. Improvements in the graph encoder and the retrieval strategy further increased robustness, especially in cases where multiple semantically similar descriptions exist. Overall, the system achieves stable performance without extensive hyperparameter tuning, suggesting that results are driven by principled architectural choices rather than overfitting.

Without any use of external data, we achieved a performance of 0.644 on the hidden test set.
