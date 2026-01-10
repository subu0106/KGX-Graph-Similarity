# Graph Similarity Analysis for Knowledge Graph Comparison

This project compares knowledge graph triples using four different similarity methods to evaluate semantic and structural similarity between graph pairs.

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
2. Applies **Agglomerative Clustering** to group semantically similar labels (threshold: 0.35)
3. Relabels graphs using cluster representatives
4. Applies **Weisfeiler-Lehman kernel** for structural similarity

**Strengths:**
- Handles paraphrases and synonyms ("play_a_role_in" ≈ "play_role_in")
- Groups related concepts ("parameters" ≈ "markers")
- Best overall performance (76.6% perfect matches)

**Example:**
```
Triple 1: ["Mitochondria", "play_a_role_in", "cell death"]
Triple 2: ["Mitochondria", "play_role_in", "cell death"]
→ KEA clusters "play_a_role_in" and "play_role_in" as synonyms
→ Similarity: 1.0 (perfect match)
```

---

### 2. TransE (Translation Embedding)

**How it works:**
1. Learns embeddings for entities and relations
2. Models relationships as translations: **h + r ≈ t**
3. Minimizes distance ||h + r - t|| for valid triples
4. Graph similarity = cosine similarity of averaged entity embeddings

**Strengths:**
- Captures relational patterns
- Works well when exact structure matches
- Fast training

**Limitations:**
- Sensitive to label variations
- Lower performance on paraphrased content (53.2% high similarity)

**Mathematical Model:**
```
Score function: f(h,r,t) = ||h + r - t||₂
Training: Minimize margin ranking loss with negative sampling
```

---

### 3. RotatE (Rotation Embedding)

**How it works:**
1. Represents relations as **rotations in complex space**
2. Models relationships as: **h ∘ r = t** (∘ = element-wise rotation)
3. Uses complex-valued embeddings (real + imaginary parts)
4. Better at capturing symmetric and antisymmetric patterns

**Strengths:**
- Handles complex relational patterns
- Better than TransE for this dataset (55.3% high similarity)
- Captures relation composition

**Limitations:**
- Still sensitive to lexical variations
- More complex than TransE

**Mathematical Model:**
```
h, t ∈ ℂᵈ (complex vectors)
r ∈ [0, 2π)ᵈ (rotation angles)
Score: ||h ∘ e^(ir) - t||
```

---

### 4. Pure WL Kernel (Structural Only)

**How it works:**
1. Creates NetworkX graphs directly from triples
2. Applies **Weisfeiler-Lehman kernel** without any preprocessing
3. Compares only graph structure and exact label matches
4. No semantic understanding

**Strengths:**
- Pure structural comparison baseline
- Fast computation
- No external dependencies on language models

**Limitations:**
- Cannot handle synonyms or paraphrases
- Lowest overall performance (53.2% perfect matches)
- Treats "play_a_role_in" and "play_role_in" as completely different

**Example:**
```
Triple 1: ["parameters", "useful for", "diagnosis"]
Triple 2: ["markers", "useful for", "diagnosis"]
→ Pure WL sees "parameters" ≠ "markers"
→ Similarity: 0.0 (no match)
```

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

## Output Files

### 1. Results CSV

**File:** `multi_method_similarity_results.csv`

**Columns:**
- `question` - Original question
- `gemini_answer_1` - First triple
- `gemini_answer_2` - Second triple
- `kea_similarity` - KEA score (0-1)
- `transe_similarity` - TransE score (can be negative)
- `rotate_similarity` - RotatE score (can be negative)
- `wl_kernel_similarity` - Pure WL score (0-1)

**Example:**
```csv
question,gemini_answer_1,gemini_answer_2,kea_similarity,transe_similarity,rotate_similarity,wl_kernel_similarity
"Question?",['A','B','C'],['A','B','C'],1.0,1.0,1.0,1.0
```

### 2. Plot Files (in `plots/` folder)

**Individual Method Plots (4 files):**
- `multi_method_similarity_results_kea_similarity_plot.png`
  - Left: Line plot of KEA scores across all questions
  - Right: Distribution histogram with mean/median

- `multi_method_similarity_results_transe_similarity_plot.png`
- `multi_method_similarity_results_rotate_similarity_plot.png`
- `multi_method_similarity_results_wl_kernel_similarity_plot.png`

**Comprehensive Comparison (1 file):**
- `multi_method_similarity_results_comparison.png`
  - Top-left: All methods overlaid
  - Top-right: Box plots comparing distributions
  - Bottom-left: Bar chart of average scores
  - Bottom-right: KEA vs Pure WL comparison

---

## Interpreting Results

### Similarity Scores

| Score | Interpretation |
|-------|----------------|
| **1.0** | Perfect match (identical or semantically equivalent) |
| **0.8-0.99** | High similarity (very similar with minor differences) |
| **0.5-0.79** | Medium similarity (some overlap, notable differences) |
| **0.2-0.49** | Low similarity (few commonalities) |
| **0.0-0.19** | Very low similarity (mostly different) |
| **Negative** | Strong dissimilarity (only for TransE/RotatE) |

### Method Comparison

**Best for:**
- **KEA:** Overall best performance, handles paraphrases
- **TransE:** Fast, good for exact structural matches
- **RotatE:** Complex relational patterns
- **Pure WL:** Baseline structural comparison

**When KEA >> Pure WL:**
This indicates the **value of semantic clustering**. Large differences show cases where synonyms/paraphrases are crucial.

---

## Performance Benchmarks

Based on 47 valid triple pairs:

| Method | Avg | Perfect (1.0) | High (≥0.8) |
|--------|-----|---------------|-------------|
| **KEA** | 79.08% | 76.6% | 76.6% |
| **RotatE** | 63.69% | 53.2% | 55.3% |
| **TransE** | 55.85% | 53.2% | 53.2% |
| **Pure WL** | 56.38% | 53.2% | 53.2% |

**Key Insight:**
KEA outperforms Pure WL by >30% in 12 cases, demonstrating the importance of semantic preprocessing.

---

## Example Output

### Console Output:
```
Multi-Method Graph Similarity Comparison

Processing row 1...
Triple 1: [['Mitochondria', 'play_a_role_in', 'cell death']]
Triple 2: [['Mitochondria', 'play_role_in', 'cell death']]

1. Calculating KEA similarity (semantic + WL)...
   KEA similarity: 1.0000

2. Calculating TransE similarity...
   TransE similarity: 0.3617

3. Calculating RotatE similarity...
   RotatE similarity: 0.3757

4. Calculating Pure WL Kernel similarity (structural only)...
   Pure WL Kernel similarity: 0.0833

Generating individual plots for each method...
Saving plots to: .../plots
  ✓ Saved: multi_method_similarity_results_kea_similarity_plot.png
  ✓ Saved: multi_method_similarity_results_transe_similarity_plot.png
  ✓ Saved: multi_method_similarity_results_rotate_similarity_plot.png
  ✓ Saved: multi_method_similarity_results_wl_kernel_similarity_plot.png
  ✓ Saved: multi_method_similarity_results_comparison.png

SUMMARY STATISTICS

KEA (Semantic Clustering + WL Kernel):
  Average:  79.08%
  Perfect (≈1.0): 36/47 (76.6%)
```

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

---

## Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "SBERT model download failed"
- First run downloads ~420MB model
- Ensure stable internet connection
- Model cached at: `~/.cache/torch/sentence_transformers/`

### Issue: "Out of memory"
- Reduce batch size in embedding training
- Process fewer rows at a time
- Use smaller embedding dimensions

### Issue: "Plots not showing"
- Check `plots/` folder in same directory as script
- Verify matplotlib is installed: `pip install matplotlib`

---

## Citation

If you use this code, please cite:

```bibtex
@software{graph_similarity_analysis,
  title = {Multi-Method Graph Similarity Analysis},
  year = {2024},
  description = {Comparison of knowledge graph similarity using KEA, TransE, RotatE, and WL Kernel methods}
}
```

### Method References

- **SBERT:** Reimers & Gurevych (2019) - Sentence-BERT
- **TransE:** Bordes et al. (2013) - Translating Embeddings for Modeling Multi-relational Data
- **RotatE:** Sun et al. (2019) - RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
- **Weisfeiler-Lehman:** Shervashidze et al. (2011) - Weisfeiler-Lehman Graph Kernels

---

## License

This project is for research and educational purposes.

---

## Contact

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Last Updated:** December 2024
