# Graph Similarity Analysis for Knowledge Graph Comparison

**Experimental Research Project**

This project explores and compares knowledge graph triples using four different similarity methods to evaluate semantic and structural similarity between graph pairs.

## Overview

Given two sets of knowledge graph triples (subject-predicate-object), this system calculates similarity scores using four complementary approaches:

1. **KEA (Knowledge Enhanced Analysis)** - Semantic clustering + Weisfeiler-Lehman kernel
2. **TransE** - Translation-based graph embeddings
3. **RotatE** - Rotation-based graph embeddings
4. **Pure WL Kernel** - Structural graph kernel without semantic preprocessing

---

## The Four Methods Explained

### 1. KEA (Semantic Clustering + WL Kernel)

**How it works:**
1. Uses **SBERT** (Sentence-BERT) to create semantic embeddings of all labels
2. Applies **Agglomerative Clustering** to group semantically similar labels
3. Relabels graphs using cluster representatives
4. Applies **Weisfeiler-Lehman kernel** for structural similarity

### 2. TransE (Translation Embedding)

**How it works:**
1. Learns embeddings for entities and relations
2. Models relationships as translations: **h + r ≈ t**
3. Minimizes distance ||h + r - t|| for valid triples
4. Graph similarity = cosine similarity of averaged entity embeddings

### 3. RotatE (Rotation Embedding)

**How it works:**
1. Represents relations as **rotations in complex space**
2. Models relationships as: **h ∘ r = t** (∘ = element-wise rotation)
3. Uses complex-valued embeddings (real + imaginary parts)
4. Captures symmetric and antisymmetric patterns

### 4. Pure WL Kernel (Structural Only)

**How it works:**
1. Creates NetworkX graphs directly from triples
2. Applies **Weisfeiler-Lehman kernel** without any preprocessing
3. Compares only graph structure and exact label matches
4. No semantic understanding

---

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `networkx` - Graph creation
- `torch` - PyTorch for embeddings
- `scikit-learn` - Machine learning utilities
- `sentence-transformers` - SBERT embeddings
- `grakel` - Graph kernel library

---

## Usage

### Input Data Format

The input CSV file (`fine_gemini_temperature.csv`) should have three columns:

```csv
question,gemini_answer_1,gemini_answer_2
"Question text?",['Subject', 'Predicate', 'Object'],['Subject', 'Predicate', 'Object']
```

**Example:**
```csv
question,gemini_answer_1,gemini_answer_2
"Do mitochondria play a role?",['Mitochondria', 'play_a_role_in', 'cell death'],['Mitochondria', 'play_role_in', 'cell death']
```

### Running the Analysis

```bash
python multi_method_similarity.py
```

### What Happens:

1. **Processing:** Analyzes all triple pairs (takes ~3-5 minutes for 50 rows)
2. **Training:** Trains TransE and RotatE models for each pair
3. **Calculation:** Computes all 4 similarity scores
4. **Output:**
   - CSV file with results
   - 5 visualization plots in `plots/` folder
   - Summary statistics in console

---

## Code Structure

### Main Files

- **`multi_method_similarity.py`** - Main script (all 4 methods + plotting)
- **`KEA/comparison.py`** - KEA implementation (SBERT + clustering + WL kernel)
- **`data/fine_gemini_temperature.csv`** - Input dataset (50 question-answer pairs)

### Key Functions

```python
# Main processing
process_dataset(input_file, output_file)
  → Returns: List of results with all 4 similarity scores

# Individual methods
calculate_similarity(triples1, triples2)  # KEA from KEA/comparison.py
calculate_transe_similarity(triples1, triples2)
calculate_rotate_similarity(triples1, triples2)
calculate_pure_wl_kernel_similarity(triples1, triples2)

# Visualization
plot_individual_method_results(results, output_file_base)
  → Creates 5 plots in plots/ folder
```

---

## Customization

### Adjust Embedding Dimensions

```python
# In multi_method_similarity.py
embedding_calculator = GraphEmbeddingSimilarity(embedding_dim=100)  # Default: 50
```

### Change Training Epochs

```python
# In GraphEmbeddingSimilarity.train_model()
def train_model(self, model, triples, epochs=200, lr=0.01):  # Default: 100
```

### Modify Clustering Threshold

```python
# In KEA/comparison.py, create_label_clusters()
clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.3,  # Default: 0.35
    metric='cosine'
)
```

## References

### Method References

- **SBERT:** Reimers & Gurevych (2019) - Sentence-BERT
- **TransE:** Bordes et al. (2013) - Translating Embeddings for Modeling Multi-relational Data
- **RotatE:** Sun et al. (2019) - RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
- **Weisfeiler-Lehman:** Shervashidze et al. (2011) - Weisfeiler-Lehman Graph Kernels

---

## Note

This is an ongoing research project for exploring graph similarity methods. The code and results are experimental and subject to change as research progresses.
