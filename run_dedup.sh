#!/bin/bash

# Data Attribute Deduplication Shell Script
# -----------------------------------------

# Default configuration
UNIQUE_DATA="./data/unique_attributes.csv"
NEW_DATA="./data/new_attributes.csv"
PROVIDER="huggingface"
MODEL_PATH="all-MiniLM-L6-v2"
OUTPUT_DIR="./output"
SHOULD_SAVE_EMBEDDINGS="false"
SYMBOLIC_WEIGHT="0.3"
SEMANTIC_WEIGHT="0.7"
TOP_K="3"

# Display help
show_help() {
  echo "Usage: ./run_dedup.sh [options]"
  echo ""
  echo "Options:"
  echo "  --unique_data PATH      Path to CSV file containing unique data attributes"
  echo "                          Default: $UNIQUE_DATA"
  echo "  --new_data PATH         Path to CSV file containing new data attributes to match"
  echo "                          Default: $NEW_DATA"
  echo "  --provider TYPE         Embedding model provider: huggingface or azure_openai"
  echo "                          Default: $PROVIDER"
  echo "  --model_path NAME       Model path for HuggingFace or model name for Azure OpenAI"
  echo "                          Default: $MODEL_PATH"
  echo "  --output_dir PATH       Directory to save output files"
  echo "                          Default: $OUTPUT_DIR"
  echo "  --save_embeddings       Save embeddings to disk (sets SHOULD_SAVE_EMBEDDINGS=true)"
  echo "  --symbolic_weight NUM   Weight for symbolic matching (0-1)"
  echo "                          Default: $SYMBOLIC_WEIGHT"
  echo "  --semantic_weight NUM   Weight for semantic matching (0-1)"
  echo "                          Default: $SEMANTIC_WEIGHT"
  echo "  --top_k NUM             Number of top matches to return"
  echo "                          Default: $TOP_K"
  echo "  --help                  Show this help message"
  echo ""
  echo "Example:"
  echo "  ./run_dedup.sh --provider azure_openai --model_path text-embedding-ada-002 --save_embeddings"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --unique_data)
      UNIQUE_DATA="$2"
      shift 2
      ;;
    --new_data)
      NEW_DATA="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save_embeddings)
      SHOULD_SAVE_EMBEDDINGS="true"
      shift
      ;;
    --symbolic_weight)
      SYMBOLIC_WEIGHT="$2"
      shift 2
      ;;
    --semantic_weight)
      SEMANTIC_WEIGHT="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    --help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Display configuration
echo "Data Attribute Deduplication"
echo "==========================="
echo "Configuration:"
echo "  Unique Data:      $UNIQUE_DATA"
echo "  New Data:         $NEW_DATA"
echo "  Provider:         $PROVIDER"
echo "  Model:            $MODEL_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Save Embeddings:  $SAVE_EMBEDDINGS"
echo "  Symbolic Weight:  $SYMBOLIC_WEIGHT"
echo "  Semantic Weight:  $SEMANTIC_WEIGHT"
echo "  Top K Matches:    $TOP_K"
echo ""

# Check if required files exist
if [ ! -f "$UNIQUE_DATA" ]; then
  echo "Error: Unique data file not found: $UNIQUE_DATA"
  exit 1
fi

if [ ! -f "$NEW_DATA" ]; then
  echo "Error: New data file not found: $NEW_DATA"
  exit 1
fi

# Export environment variables for the Python script
export UNIQUE_DATA
export NEW_DATA
export PROVIDER
export MODEL_PATH
export OUTPUT_DIR
export SHOULD_SAVE_EMBEDDINGS
export SYMBOLIC_WEIGHT
export SEMANTIC_WEIGHT
export TOP_K

# Run the Python script
echo "Starting deduplication process..."
cd "$(dirname "$0")" # Change to script directory
python -m src.main

# Check if the script executed successfully
if [ $? -eq 0 ]; then
  echo ""
  echo "Deduplication completed successfully!"
  echo "Results saved to: $OUTPUT_DIR/results/"
else
  echo ""
  echo "Error: Deduplication process failed."
  exit 1
fi
