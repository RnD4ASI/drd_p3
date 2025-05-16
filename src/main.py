#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import json
import datetime
from loguru import logger

# Configure loguru
# Remove default handler
logger.remove()
# Add a new handler with desired format
logger.add(
    lambda msg: print(msg, end=''),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Local imports
from src.hybrid_search import HybridSearch, process_attribute_data, generate_embeddings
from src.utility import DataUtility
from src.embedding import EmbeddingModel, concatenate_attribute_data, save_embeddings, load_embeddings

def main():
    """Main function to run the data attribute deduplication pipeline."""
    # Read parameters from environment variables with defaults
    unique_data = os.environ.get('UNIQUE_DATA', './data/unique_attributes.csv')
    new_data = os.environ.get('NEW_DATA', './data/new_attributes.csv')
    provider = os.environ.get('PROVIDER', 'huggingface')
    model_path = os.environ.get('MODEL_PATH', 'all-MiniLM-L6-v2')
    output_dir = os.environ.get('OUTPUT_DIR', './output')
    should_save_embeddings = os.environ.get('SHOULD_SAVE_EMBEDDINGS', 'false').lower() in ('true', 'yes', '1')
    symbolic_weight = float(os.environ.get('SYMBOLIC_WEIGHT', '0.3'))
    semantic_weight = float(os.environ.get('SEMANTIC_WEIGHT', '0.7'))
    top_k = int(os.environ.get('TOP_K', '3'))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped output subdirectories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f'run_{timestamp}')
    embeddings_dir = os.path.join(run_dir, 'embeddings')
    results_dir = os.path.join(run_dir, 'results')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Log the output directories
    logger.info(f"Output will be saved to: {run_dir}")
    logger.info(f"Embeddings will be saved to: {embeddings_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    logger.info(f"Starting data attribute deduplication with {provider} model: {model_path}")
    
    # Initialize utilities
    data_util = DataUtility()
    hybrid_search = HybridSearch(symbolic_weight=symbolic_weight, semantic_weight=semantic_weight)
    
    try:
        # 1. Read CSV files
        logger.info(f"Reading unique data attributes from {unique_data}")
        unique_df = data_util.text_operation('load', unique_data, file_type='csv')
        logger.info(f"Loaded {len(unique_df)} unique data attributes")
        
        logger.info(f"Reading new data attributes from {new_data}")
        new_df = data_util.text_operation('load', new_data, file_type='csv')
        logger.info(f"Loaded {len(new_df)} new data attributes")
        
        # 2. Process data attributes
        unique_attributes = process_attribute_data(unique_df)
        new_attributes = process_attribute_data(new_df)
        
        # 3. Generate embeddings for unique attributes
        logger.info(f"Generating embeddings for unique attributes using {provider}")
        unique_embeddings = generate_embeddings(
            unique_attributes,
            model_path=model_path,
            provider=provider
        )
        logger.info(f"Generated {len(unique_embeddings)} embeddings for unique attributes")
        
        # Save embeddings if requested
        if should_save_embeddings:
            unique_embedding_path = save_embeddings(
                unique_embeddings,
                embeddings_dir,
                metadata={
                    'type': 'unique_attributes',
                    'count': len(unique_attributes),
                    'model': model_path,
                    'provider': provider
                }
            )
            logger.info(f"Saved unique attribute embeddings to {unique_embedding_path}")
        
        # 4. Generate embeddings for new attributes
        logger.info(f"Generating embeddings for new attributes using {provider}")
        new_embeddings = generate_embeddings(
            new_attributes,
            model_path=model_path,
            provider=provider
        )
        logger.info(f"Generated {len(new_embeddings)} embeddings for new attributes")
        
        # Save embeddings if requested
        if should_save_embeddings:
            new_embedding_path = save_embeddings(
                new_embeddings,
                embeddings_dir,
                metadata={
                    'type': 'new_attributes',
                    'count': len(new_attributes),
                    'model': model_path,
                    'provider': provider
                }
            )
            logger.info(f"Saved new attribute embeddings to {new_embedding_path}")
        
        # 5. Apply hybrid search to find matches
        logger.info(f"Finding top {top_k} matches for each new attribute")
        
        results = []
        for i, new_attr in enumerate(new_attributes):
            # Find top matches
            matches = hybrid_search.find_top_matches(
                new_attr,
                new_embeddings[i],
                unique_attributes,
                unique_embeddings,
                top_k=top_k
            )
            
            # Add to results
            result = {
                'id': new_attr['id'],
                'name': new_attr['name'],
                'definition': new_attr['definition'],
                'matches': matches
            }
            results.append(result)
        
        # 6. Save results
        results_file = os.path.join(results_dir, 'hybrid_search_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved hybrid search results to {results_file}")
        
        # 7. Create summary DataFrame
        summary_rows = []
        for result in results:
            for match_idx, match in enumerate(result['matches']):
                summary_rows.append({
                    'new_id': result['id'],
                    'new_name': result['name'],
                    'new_definition': result['definition'],
                    'match_rank': match_idx + 1,
                    'match_id': match['candidate_id'],
                    'match_name': match['candidate_name'],
                    'match_definition': match['candidate_definition'],
                    'symbolic_score': match['symbolic_score'],
                    'semantic_score': match['semantic_score'],
                    'hybrid_score': match['hybrid_score']
                })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(results_dir, 'hybrid_search_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary to {summary_file}")
        
        logger.info("Data attribute deduplication completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data attribute deduplication: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
