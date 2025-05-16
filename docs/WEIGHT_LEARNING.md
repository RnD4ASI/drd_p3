# Weight Learning for Hybrid Search

This document explains how to learn optimal weights for combining symbolic and semantic scores in the hybrid search functionality.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Data Requirements](#data-requirements)
3. [Running the Weight Learning](#running-the-weight-learning)
4. [Understanding the Output](#understanding-the-output)
5. [Applying Learned Weights](#applying-learned-weights)
6. [Advanced Configuration](#advanced-configuration)

## Prerequisites

- Python 3.7+
- Required Python packages (install with `pip install -r requirements.txt`):
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - loguru

## Data Requirements

### Input Data Format

The weight learning script expects a CSV file with the following columns:
- `new_id`: Identifier for the new data attribute
- `new_name`: Name of the new data attribute
- `new_definition`: Definition/description of the new data attribute
- `match_rank`: Rank of the match (1 = best match, higher numbers = worse matches)
- `match_id`: Identifier for the matched existing attribute
- `match_name`: Name of the matched existing attribute
- `match_definition`: Definition of the matched existing attribute
- `symbolic_score`: Symbolic similarity score (0-1)
- `semantic_score`: Semantic similarity score (0-1)
- `hybrid_score`: Combined score (not used in training)

### Data Location

By default, the script looks for data at:
```
./output/run_<timestamp>/results/hybrid_search_summary.csv
```

You can specify a custom path using the `--data_path` argument.

## Running the Weight Learning

### Basic Usage

1. Make sure you have run the deduplication process to generate the training data
2. Run the weight learning script:

```bash
# Make the script executable if needed
chmod +x run_learn_weights.sh

# Run with default settings
./run_learn_weights.sh

# Or specify a custom data path
./run_learn_weights.sh --data_path /path/to/your/hybrid_search_summary.csv
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to the input CSV file | `./output/run_<latest>/results/hybrid_search_summary.csv` |
| `--output_dir` | Directory to save results | `./output/learned_weights` |
| `--test_size` | Fraction of data to use for testing | 0.2 |
| `--random_state` | Random seed for reproducibility | 42 |
| `--n_alphas` | Number of alpha values to try | 100 |

## Understanding the Output

After running the script, you'll find the following files in the output directory:

1. `weights.json` - Contains the learned weights in JSON format:
   ```json
   {
     "symbolic_weight": 0.35,
     "semantic_weight": 0.65,
     "average_precision": 0.95
   }
   ```

2. `learning_curve.png` - A plot showing how the model performance varies with different alpha values.

3. `training_data.csv` - A copy of the training data used for reference.

### Interpreting the Results

- **symbolic_weight**: The optimal weight for the symbolic similarity score
- **semantic_weight**: The optimal weight for the semantic similarity score
- **average_precision**: The model's performance metric (higher is better, max 1.0)

The learning curve plot shows:
- X-axis: Alpha value (weight for semantic score)
- Y-axis: Average Precision score
- Vertical dashed line: The optimal alpha value

## Applying Learned Weights

To use the learned weights in your hybrid search:

1. After running the weight learning, you'll find the weights in `output/learned_weights/weights.json`

2. Update your code to use these weights when initializing the `HybridSearch` class:

```python
import json
from src.hybrid_search import HybridSearch

# Load learned weights
with open('output/learned_weights/weights.json', 'r') as f:
    weights = json.load(f)

# Initialize HybridSearch with learned weights
hybrid_search = HybridSearch(
    symbolic_weight=weights['symbolic_weight'],
    semantic_weight=weights['semantic_weight']
)
```

Or update your configuration file/script to use these weights.

## Advanced Configuration

### Customizing the Training

You can modify the `TrainingConfig` class in `src/learn_weights.py` to adjust:

- `min_alpha`/`max_alpha`: Range of alpha values to search
- `n_splits`: Number of cross-validation folds
- `test_size`: Fraction of data to use for testing

### Custom Evaluation Metrics

By default, the script uses Average Precision as the evaluation metric. You can modify the `evaluate_alpha` method in the `WeightLearner` class to use other metrics like:
- Precision@K
- Recall@K
- F1-score
- ROC-AUC

### Hyperparameter Tuning

For more advanced tuning, you could:
1. Add grid search over multiple hyperparameters
2. Implement cross-validation
3. Add early stopping
4. Include class weights for imbalanced data

## Troubleshooting

- **No positive examples**: Ensure your input data has some rows where `match_rank == 1`
- **Low performance**: Try collecting more training data or adjusting the alpha range
- **Memory issues**: Reduce `n_alphas` or process the data in batches

## Best Practices

1. **Data Quality**: Ensure your training data is clean and representative
2. **Validation**: Always validate on a held-out test set
3. **Versioning**: Keep track of which weights were used for which model versions
4. **Retraining**: Periodically retrain the weights as you collect more data
