import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from sentence_transformers import util
from loguru import logger

# Local imports
from src.embedding import EmbeddingModel, concatenate_attribute_data
from src.utility import DataUtility

# logger is imported from loguru

class HybridSearch:
    """Class to perform hybrid searching of data attributes using both symbolic and semantic methods."""
    
    def __init__(self, symbolic_weight: float = 0.3, semantic_weight: float = 0.7):
        """Initialize the HybridSearch class.
        
        Parameters:
            symbolic_weight (float): Weight for symbolic matching (0-1)
            semantic_weight (float): Weight for semantic matching (0-1)
        """
        if not np.isclose(symbolic_weight + semantic_weight, 1.0):
            logger.warning(f"Weights do not sum to 1.0: symbolic={symbolic_weight}, semantic={semantic_weight}")
            # Normalize weights
            total = symbolic_weight + semantic_weight
            symbolic_weight /= total
            semantic_weight /= total
            logger.info(f"Normalized weights: symbolic={symbolic_weight}, semantic={semantic_weight}")
        
        self.symbolic_weight = symbolic_weight
        self.semantic_weight = semantic_weight
        self.data_utility = DataUtility()
        logger.info(f"HybridSearch initialized with weights: symbolic={self.symbolic_weight}, semantic={self.semantic_weight}")
    
    def string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using sequence matcher.
        
        Parameters:
            str1 (str): First string
            str2 (str): Second string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0.0
            
        # Convert to lowercase for case-insensitive matching
        str1 = str1.lower()
        str2 = str2.lower()
        
        # Use SequenceMatcher for string similarity
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for symbolic matching.
        
        Parameters:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def symbolic_match_score(self, item1: Dict[str, str], item2: Dict[str, str]) -> float:
        """Calculate symbolic match score between two data attributes.
        
        Parameters:
            item1 (Dict[str, str]): First data attribute with 'name' and 'definition' keys
            item2 (Dict[str, str]): Second data attribute with 'name' and 'definition' keys
            
        Returns:
            float: Symbolic match score between 0 and 1
        """
        # Extract name and definition
        name1 = item1.get('name', '')
        name2 = item2.get('name', '')
        def1 = item1.get('definition', '')
        def2 = item2.get('definition', '')
        
        # Preprocess texts
        name1 = self._preprocess_text(name1)
        name2 = self._preprocess_text(name2)
        def1 = self._preprocess_text(def1)
        def2 = self._preprocess_text(def2)
        
        # Calculate name and definition similarities
        name_sim = self.string_similarity(name1, name2)
        def_sim = self.string_similarity(def1, def2)
        
        # Equal weight for name and definition similarity
        return (name_sim + def_sim) / 2
    
    def semantic_match_score(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate semantic match score using cosine similarity of embeddings.
        
        Parameters:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Semantic match score between 0 and 1
        """
        # Ensure embeddings are 1D arrays
        if embedding1.ndim > 1:
            embedding1 = embedding1.flatten()
        if embedding2.ndim > 1:
            embedding2 = embedding2.flatten()
            
        # Reshape for cosine_similarity
        embedding1_r = embedding1.reshape(1, -1)
        embedding2_r = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1_r, embedding2_r)[0][0]
        
        # Ensure result is between 0 and 1
        return max(0, min(1, float(similarity)))
    
    def hybrid_match_score(self, item1: Dict[str, str], item2: Dict[str, str], 
                          embedding1: np.ndarray, embedding2: np.ndarray) -> Dict[str, float]:
        """Calculate hybrid match score combining symbolic and semantic methods.
        
        Parameters:
            item1 (Dict[str, str]): First data attribute with 'name' and 'definition' keys
            item2 (Dict[str, str]): Second data attribute with 'name' and 'definition' keys
            embedding1 (np.ndarray): Embedding vector for first attribute
            embedding2 (np.ndarray): Embedding vector for second attribute
            
        Returns:
            Dict[str, float]: Dictionary with symbolic, semantic, and hybrid scores
        """
        # Calculate symbolic score
        symbolic_score = self.symbolic_match_score(item1, item2)
        
        # Calculate semantic score
        semantic_score = self.semantic_match_score(embedding1, embedding2)
        
        # Calculate weighted hybrid score
        hybrid_score = (self.symbolic_weight * symbolic_score) + (self.semantic_weight * semantic_score)
        
        return {
            'symbolic_score': symbolic_score,
            'semantic_score': semantic_score,
            'hybrid_score': hybrid_score
        }
    
    def find_top_matches(self, query_item: Dict[str, Any], query_embedding: np.ndarray,
                       candidates: List[Dict[str, Any]], candidate_embeddings: np.ndarray,
                       top_k: int = 3) -> List[Dict[str, Any]]:
        """Find top matches for a query item among candidates.
        
        Parameters:
            query_item (Dict[str, Any]): Query data attribute
            query_embedding (np.ndarray): Query embedding
            candidates (List[Dict[str, Any]]): List of candidate data attributes
            candidate_embeddings (np.ndarray): Embeddings for candidates
            top_k (int): Number of top matches to return
            
        Returns:
            List[Dict[str, Any]]: List of top matches with scores
        """
        results = []
        
        # Calculate match scores for all candidates
        for i, candidate in enumerate(candidates):
            # Skip if comparing with itself (by ID)
            if 'id' in query_item and 'id' in candidate and query_item['id'] == candidate['id']:
                continue
                
            # Get candidate embedding
            candidate_embedding = candidate_embeddings[i]
            
            # Calculate hybrid score
            scores = self.hybrid_match_score(
                {'name': query_item['name'], 'definition': query_item['definition']},
                {'name': candidate['name'], 'definition': candidate['definition']},
                query_embedding,
                candidate_embedding
            )
            
            # Add to results
            results.append({
                'candidate_id': candidate.get('id', f"candidate_{i}"),
                'candidate_name': candidate['name'],
                'candidate_definition': candidate['definition'],
                'symbolic_score': scores['symbolic_score'],
                'semantic_score': scores['semantic_score'],
                'hybrid_score': scores['hybrid_score']
            })
        
        # Sort by hybrid score in descending order
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Return top k results
        return results[:top_k]


def process_attribute_data(data_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process data attribute dataframe into a list of dictionaries.
    
    Parameters:
        data_df (pd.DataFrame): Dataframe containing data attributes
        
    Returns:
        List[Dict[str, Any]]: List of data attributes as dictionaries
    """
    # Make sure required columns exist
    required_cols = ['id', 'name', 'definition']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Input dataframe missing required columns: {missing_cols}")
    
    # Convert dataframe to list of dictionaries
    data_list = []
    for _, row in data_df.iterrows():
        data_list.append({
            'id': str(row['id']),
            'name': str(row['name']),
            'definition': str(row['definition'])
        })
    
    return data_list


def generate_embeddings(data_list: List[Dict[str, Any]], 
                       model_path: str, 
                       provider: str = "huggingface") -> np.ndarray:
    """Generate embeddings for data attributes.
    
    Parameters:
        data_list (List[Dict[str, Any]]): List of data attributes
        model_path (str): Path to the embedding model
        provider (str): Model provider, either 'huggingface' or 'azure_openai'
        
    Returns:
        np.ndarray: Array of embedding vectors
    """
    # Initialize embedding model
    embedding_model = EmbeddingModel(model_path=model_path, provider=provider)
    embedding_model.load_model()
    
    # Prepare text input for embedding
    texts = []
    for item in data_list:
        combined_text = concatenate_attribute_data(item['name'], item['definition'])
        texts.append(combined_text)
    
    # Generate embeddings
    embeddings = embedding_model.get_embeddings(texts)
    
    return embeddings
