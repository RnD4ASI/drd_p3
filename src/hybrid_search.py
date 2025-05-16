import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from sklearn.feature_extraction import text as sklearn_text
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
    
    def _preprocess_text(self, name: str, definition: str) -> str:
        """
        Concatenate, lowercase, remove punctuation/stopwords from name and definition.
        """
        stop_words = set(sklearn_text.ENGLISH_STOP_WORDS)
        text = f"{name} {definition}".lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [w for w in text.split() if w not in stop_words]
        return ' '.join(tokens)
    
    def symbolic_scores(self, query_item: Dict[str, str], pool_items: list) -> list:
        """
        Compute symbolic (TF-IDF cosine similarity) scores between query and pool.
        Returns a list of floats, one per pool item.
        """
        # Preprocess all texts
        pool_docs = [self._preprocess_text(item.get('name',''), item.get('definition','')) for item in pool_items]
        query_doc = self._preprocess_text(query_item.get('name',''), query_item.get('definition',''))
        all_docs = pool_docs + [query_doc]
        # Fit TF-IDF
        vectorizer = TfidfVectorizer().fit(all_docs)
        tfidf_matrix = vectorizer.transform(all_docs)
        query_vec = tfidf_matrix[-1]
        pool_matrix = tfidf_matrix[:-1]
        # Cosine similarity between query and all pool
        sims = cosine_similarity(query_vec, pool_matrix)[0]
        return sims.tolist()
    
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
    
    def hybrid_match_scores(self, query_item: Dict[str, str], pool_items: list, query_embedding: np.ndarray, pool_embeddings: np.ndarray) -> list:
        """
        Compute hybrid scores for a query against a pool.
        Returns a list of dicts: [{symbolic_score, semantic_score, hybrid_score}, ...]
        """
        # Symbolic (TF-IDF cosine) scores
        symbolic_scores = self.symbolic_scores(query_item, pool_items)
        # Semantic (embedding cosine) scores
        semantic_scores = [self.semantic_match_score(query_embedding, emb) for emb in pool_embeddings]
        # Hybrid scores
        results = []
        for sym, sem in zip(symbolic_scores, semantic_scores):
            hybrid = self.symbolic_weight * sym + self.semantic_weight * sem
            results.append({'symbolic_score': sym, 'semantic_score': sem, 'hybrid_score': hybrid})
        return results
    
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
        # Compute all hybrid scores in batch
        scores_list = self.hybrid_match_scores(query_item, candidates, query_embedding, candidate_embeddings)
        for i, (candidate, scores) in enumerate(zip(candidates, scores_list)):
            # Skip if comparing with itself (by ID)
            if 'id' in query_item and 'id' in candidate and query_item['id'] == candidate['id']:
                continue
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
