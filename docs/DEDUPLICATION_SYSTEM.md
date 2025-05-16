# Data Attribute Deduplication System

This document provides a comprehensive guide to using the data attribute deduplication system, which combines symbolic and semantic matching to identify duplicate or similar data attributes across datasets.

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Data Requirements](#data-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Detailed Usage](#detailed-usage)
7. [System Design](#system-design)
8. [Output Format](#output-format)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Topics](#advanced-topics)

## System Overview

The data attribute deduplication system helps identify and match similar data attributes across different datasets using a hybrid approach that combines:

- **Symbolic Matching**: Exact or fuzzy string matching of attribute names and definitions
- **Semantic Matching**: Deep learning-based semantic understanding of attribute meanings
- **Hybrid Scoring**: Weighted combination of symbolic and semantic scores

## Prerequisites

- Python 3.7+
- pip (Python package manager)
- Git (for cloning the repository)
- Sufficient disk space for models (1-2GB recommended)

## Data Requirements

### Input Files

The system requires two CSV files:

1. **Unique Attributes File**: Contains the reference set of attributes
2. **New Attributes File**: Contains new attributes to match against the reference set

### Required Columns

Both input files must contain at least these columns:
- `id`: Unique identifier for the attribute
- `name`: Name of the attribute
- `definition`: Description or definition of the attribute

### Example Input

`unique_attributes.csv`:
```csv
id,name,definition
DA001,Customer ID,A unique identifier assigned to each customer
DA002,Product SKU,Stock keeping unit code used to identify products in inventory
...
```

`new_attributes.csv`:
```csv
id,name,definition
NA001,Client Identifier,A unique ID given to each client in the system
NA002,Product Stock Code,Code used to identify items in inventory
...
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd drd_dedup3
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download embedding models** (if not using default)
   ```bash
   # Example for downloading a specific model
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

## Quick Start

1. **Prepare your data**
   - Place your `unique_attributes.csv` and `new_attributes.csv` in the `data` directory

2. **Run the deduplication**
   ```bash
   ./run_dedup.sh --model_path all-MiniLM-L6-v2 --save_embeddings
   ```

3. **View results**
   - Results will be saved in `output/run_<timestamp>/`
   - Check `results/hybrid_search_summary.csv` for matches

## Detailed Usage

### Running with Custom Options

```bash
./run_dedup.sh \
  --unique_data ./data/your_unique_attributes.csv \
  --new_data ./data/your_new_attributes.csv \
  --model_path finance_embeddings \
  --output_dir ./custom_output \
  --symbolic_weight 0.4 \
  --semantic_weight 0.6 \
  --top_k 5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--unique_data` | Path to unique attributes CSV | `./data/unique_attributes.csv` |
| `--new_data` | Path to new attributes CSV | `./data/new_attributes.csv` |
| `--model_path` | Path to embedding model | `all-MiniLM-L6-v2` |
| `--provider` | Model provider (`huggingface` or `azure_openai`) | `huggingface` |
| `--output_dir` | Base output directory | `./output` |
| `--save_embeddings` | Save generated embeddings | `False` |
| `--symbolic_weight` | Weight for symbolic matching | `0.3` |
| `--semantic_weight` | Weight for semantic matching | `0.7` |
| `--top_k` | Number of top matches to return | `3` |

## System Design

### Core Components

1. **Embedding Model (`embedding.py`)**
   - Handles text embedding generation
   - Supports multiple model providers
   - Includes caching for performance

2. **Hybrid Search (`hybrid_search.py`)**
   - Combines symbolic and semantic matching
   - Configurable weights for each method
   - Efficient similarity search

3. **Main Pipeline (`main.py`)**
   - Orchestrates the deduplication process
   - Handles input/output operations
   - Manages resources and configurations

### Workflow

1. **Data Loading**
   - Load unique and new attributes
   - Validate input data

2. **Embedding Generation**
   - Generate embeddings for all attributes
   - Cache embeddings if enabled

3. **Matching**
   - Calculate symbolic similarities
   - Calculate semantic similarities
   - Combine scores using weights
   - Find top-k matches

4. **Output Generation**
   - Save results in multiple formats
   - Generate summary statistics
   - Save embeddings (if enabled)

## Output Format

The system generates the following output files:

### 1. Hybrid Search Results (`hybrid_search_results.json`)

Contains detailed matching results in JSON format:

```json
{
  "NA001": [
    {
      "match_id": "DA001",
      "match_name": "Customer ID",
      "match_definition": "A unique identifier assigned to each customer",
      "symbolic_score": 0.46,
      "semantic_score": 0.69,
      "hybrid_score": 0.62
    },
    ...
  ],
  ...
}
```

### 2. Summary CSV (`hybrid_search_summary.csv`)

Tabular format of results for easy analysis:

```csv
new_id,new_name,match_rank,match_id,match_name,symbolic_score,semantic_score,hybrid_score
NA001,Client Identifier,1,DA001,Customer ID,0.46,0.69,0.62
...
```

### 3. Embeddings (Optional)

If enabled, embeddings are saved in NumPy format for future use.

## Performance Considerations

### Memory Usage
- Embedding models can be memory-intensive
- For large datasets, consider:
  - Using smaller models
  - Processing in batches
  - Using a machine with more RAM

### Processing Time
- Initial model loading may take time
- Embedding generation scales with:
  - Number of attributes
  - Length of definitions
  - Model complexity

### Storage
- Embeddings can consume significant disk space
- Enable `--save_embeddings` only if needed for future use

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure the model path is correct
   - Check internet connection if downloading
   - Verify sufficient disk space

2. **Memory Errors**
   - Reduce batch size
   - Use a smaller model
   - Close other memory-intensive applications

3. **Low Match Quality**
   - Adjust symbolic/semantic weights
   - Improve attribute definitions
   - Try a different embedding model

### Logs

Check the console output or log files in the output directory for detailed error messages.

## Advanced Topics

### Custom Embedding Models

To use a custom model:

1. Place the model in the `models` directory
2. Update the model path in the configuration
3. Ensure the model is compatible with the Sentence Transformers API

### Integration with Other Systems

The system can be integrated into larger data pipelines by importing and using the core modules:

```python
from src.embedding import EmbeddingModel
from src.hybrid_search import HybridSearch

# Initialize components
embedder = EmbeddingModel(model_path="your/model/path")
searcher = HybridSearch(symbolic_weight=0.4, semantic_weight=0.6)

# Generate embeddings
embeddings = embedder.get_embeddings(["your text"])

# Perform search
results = searcher.search(query_embeddings, target_embeddings, top_k=5)
```

### Scaling Up

For very large datasets, consider:
- Using approximate nearest neighbor search (e.g., FAISS, Annoy)
- Implementing distributed processing
- Using a vector database for persistent storage

### Customization

You can extend the system by:
1. Adding new similarity metrics
2. Implementing custom preprocessing
3. Supporting additional data formats
4. Adding visualization tools

## Support

For issues or feature requests, please open an issue in the repository.
