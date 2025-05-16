import os
import numpy as np
import json
import time
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer, models, util
import torch
from loguru import logger
from src.azure_integration import AzureOpenAIClient
import tiktoken


# logger is imported from loguru

class EmbeddingModel:
    """Handles text embedding using various models."""
    
    def __init__(self, model_path: Optional[str] = None, provider: str = "huggingface"):
        """Initialize the EmbeddingModel.
        
        Parameters:
            model_path (Optional[str]): Path to the embedding model
            provider (str): Model provider, either 'huggingface' or 'azure_openai'
        """
        self.model_path = model_path
        self.provider = provider.lower()
        self.model = None
        self.tokenizer = None
        self.azure_client = None
        
        # Set device for local models
        if self.provider == "huggingface":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
        
        # Initialize Azure client if using Azure OpenAI
        if self.provider == "azure_openai":
            self.azure_client = AzureOpenAIClient()
            if not self.azure_client.is_configured:
                logger.warning("Azure OpenAI not fully configured. Will fall back to HuggingFace if Azure embedding is requested.")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load the embedding model.
        
        Parameters:
            model_path (Optional[str]): Path to the embedding model or model name
                                        Can be:
                                        - A HuggingFace model name (e.g., 'all-MiniLM-L6-v2')
                                        - A relative path to local model directory (e.g., 'models/my-model')
                                        - An absolute path to local model directory
                                        - An Azure OpenAI model name if provider is 'azure_openai'
        """
        try:
            if model_path:
                self.model_path = model_path
            
            if not self.model_path:
                raise ValueError("Model path not provided")
            
            # If using Azure OpenAI, no need to load a local model
            if self.provider == "azure_openai":
                logger.info(f"Using Azure OpenAI embedding model: {self.model_path}")
                return
            
            # Load local model for HuggingFace
            logger.info(f"Loading embedding model from {self.model_path}")
            
            # Check if this is a local path - try multiple locations
            # First try as is (absolute path)
            if os.path.exists(self.model_path):
                local_path = self.model_path
                logger.info(f"Found model at absolute path: {local_path}")
            # Then try in current directory
            elif os.path.exists(os.path.join(os.getcwd(), self.model_path)):
                local_path = os.path.join(os.getcwd(), self.model_path)
                logger.info(f"Found model in current directory: {local_path}")
            # Finally try in 'model' folder
            elif os.path.exists(os.path.join(os.getcwd(), 'model', self.model_path)):
                local_path = os.path.join(os.getcwd(), 'model', self.model_path)
                logger.info(f"Found model in model folder: {local_path}")
            # If none of the above, try as a HuggingFace model
            else:
                local_path = None
                
            if local_path:
                logger.info(f"Found local model at {local_path}")
                # Use absolute path for local models
                model_path_to_use = os.path.abspath(local_path)
                
                # Check if the directory has required model files
                required_files = ["config.json", "modules.json"]
                existing_files = [f for f in required_files if os.path.exists(os.path.join(model_path_to_use, f))]
                files_exist = len(existing_files) == len(required_files)
                
                logger.info(f"Checking for required model files in {model_path_to_use}:")
                for file in required_files:
                    file_path = os.path.join(model_path_to_use, file)
                    logger.info(f"  - {file}: {'EXISTS' if os.path.exists(file_path) else 'MISSING'}")
                
                if files_exist:
                    logger.info("✅ All required model files found. Proceeding with local loading.")
                    # First approach: Direct loading with local_files_only flag
                    try:
                        logger.info("🔄 METHOD 1: Attempting to load model using direct path with local_files_only=True")
                        start_time = time.time()
                        self.model = SentenceTransformer(model_path_to_use, local_files_only=True)
                        load_time = time.time() - start_time
                        logger.info(f"✅ METHOD 1: Successfully loaded model using direct path with local_files_only=True (took {load_time:.2f}s)")
                        return
                    except Exception as e:
                        logger.warning(f"❌ METHOD 1: Failed to load model directly: {str(e)}")
                        
                    # Second approach: Manual module construction
                    try:
                        logger.info("🔄 METHOD 2: Attempting to load model by constructing modules manually...")
                        start_time = time.time()
                        transformer = models.Transformer(model_path_to_use, local_files_only=True)
                        logger.info(f"  - Transformer module loaded with dimension: {transformer.get_word_embedding_dimension()}")
                        
                        pooling = models.Pooling(transformer.get_word_embedding_dimension(), 
                                                pooling_mode_mean_tokens=True,
                                                pooling_mode_cls_token=False, 
                                                pooling_mode_max_tokens=False)
                        logger.info("  - Pooling module loaded")
                        
                        normalize = models.Normalize()
                        logger.info("  - Normalize module loaded")
                        
                        self.model = SentenceTransformer(modules=[transformer, pooling, normalize])
                        load_time = time.time() - start_time
                        logger.info(f"✅ METHOD 2: Successfully loaded model using manual module construction (took {load_time:.2f}s)")
                        return
                    except Exception as e:
                        logger.error(f"❌ METHOD 2: Failed to load model using manual module construction: {str(e)}")
                        raise
                else:
                    logger.warning(f"❌ Required model files not found in {model_path_to_use}. Found {len(existing_files)}/{len(required_files)} required files.")
            
            # If not a local path or local files don't exist, try to download from HuggingFace
            logger.info(f"🔄 METHOD 3: Attempting to load model from HuggingFace: {self.model_path}")
            start_time = time.time()
            self.model = SentenceTransformer(self.model_path)
            load_time = time.time() - start_time
            logger.info(f"✅ METHOD 3: Successfully loaded model from HuggingFace (took {load_time:.2f}s)")
            
            # Optionally save the model locally for future use
            if local_path and not os.path.exists(local_path):
                logger.info(f"💾 Saving model to {local_path} for future use")
                start_time = time.time()
                os.makedirs(local_path, exist_ok=True)
                self.model.save(local_path)
                save_time = time.time() - start_time
                logger.info(f"✅ Model saved to {local_path} (took {save_time:.2f}s)")
                logger.info(f"📁 Model files saved to disk:")
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        logger.info(f"  - {os.path.relpath(file_path, local_path)}: {file_size:.2f} MB")
        except Exception as e:
            logger.error(f"Error in load_model: {e}")
            raise
    
    def get_embeddings(self, texts: Union[str, List[str]], batch_size: Optional[int] = None, max_tokens_per_batch: int = 8192, encoding_name: str = "cl100k_base", buffer_ratio: float = 0.9) -> np.ndarray:
        """
        Get embeddings for the given texts, batching by total token count to avoid model limits.

        Parameters:
            texts (Union[str, List[str]]): Text(s) to embed
            batch_size (Optional[int]): Maximum number of texts per batch (soft limit, actual batch may be smaller if token limit is hit). If None, will be dynamically calculated based on token counts and max_tokens_per_batch.
            max_tokens_per_batch (int): Maximum number of tokens allowed in a batch (default 8192)
            encoding_name (str): tiktoken encoding name to use (default 'cl100k_base' for OpenAI, change if using other models)
            buffer_ratio (float): Safety ratio for dynamic batch size calculation (default 0.9, i.e. use 90% of token limit)
        Returns:
            np.ndarray: Embedding vectors
        Raises:
            ImportError if tiktoken is not installed
        """
        try:
            if tiktoken is None:
                raise ImportError("tiktoken is required for token-based batching. Please install it and add to requirements.txt.")

            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]

            if not isinstance(texts, list):
                raise ValueError(f"Texts must be a string or a list of strings, got {type(texts)}")

            total_texts = len(texts)
            encoding = tiktoken.get_encoding(encoding_name)
            token_counts = [len(encoding.encode(text)) for text in texts]
            total_tokens = sum(token_counts)

            # Dynamically calculate batch_size if not provided
            if batch_size is None or batch_size <= 0:
                if total_texts > 1:
                    avg_tokens_per_text = total_tokens / total_texts
                    batch_size = max(1, int((max_tokens_per_batch * buffer_ratio) / avg_tokens_per_text))
                    logger.info(f"Dynamically calculated batch_size: {batch_size} (avg tokens/text={avg_tokens_per_text:.2f}, buffer_ratio={buffer_ratio})")
                else:
                    batch_size = 1

            logger.info(f"Generating embeddings for {total_texts} texts with max_tokens_per_batch={max_tokens_per_batch}, batch_size={batch_size}")

            # Create batches where sum(tokens) <= max_tokens_per_batch and len(batch) <= batch_size
            batches = []
            current_batch = []
            current_tokens = 0
            for idx, (text, tokens) in enumerate(zip(texts, token_counts)):
                if tokens > max_tokens_per_batch:
                    logger.error(f"Text at index {idx} exceeds max_tokens_per_batch ({tokens} > {max_tokens_per_batch}), skipping.")
                    continue  # Or raise error if you want to be strict
                if (current_tokens + tokens > max_tokens_per_batch) or (len(current_batch) >= batch_size):
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [text]
                    current_tokens = tokens
                else:
                    current_batch.append(text)
                    current_tokens += tokens
            if current_batch:
                batches.append(current_batch)

            logger.info(f"Total batches to process: {len(batches)}")
            all_embeddings = []
            for batch_idx, batch_texts in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch_texts)} texts, total tokens: {sum(len(encoding.encode(t)) for t in batch_texts)}")
                if self.provider == "azure_openai":
                    if not self.azure_client or not self.azure_client.is_configured:
                        raise ValueError("Azure OpenAI client not configured. Check environment variables.")
                    try:
                        batch_embeddings = self.azure_client.get_embeddings(batch_texts, self.model_path)
                        if isinstance(batch_embeddings[0], list):
                            all_embeddings.extend(batch_embeddings)
                        else:
                            all_embeddings.append(batch_embeddings)
                    except Exception as e:
                        logger.error(f"Azure OpenAI embeddings failed for batch {batch_idx+1}: {e}")
                        raise
                elif self.provider == "huggingface":
                    if self.model is None:
                        raise ValueError("Model not loaded. Call load_model first.")
                    if isinstance(self.model, SentenceTransformer):
                        batch_embeddings = self.model.encode(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                    else:
                        raise ValueError(f"Unsupported model type: {type(self.model)}")
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            return np.array(all_embeddings)
        except Exception as e:
            logger.error(f"Error in get_embeddings: {e}")
            raise

def concatenate_attribute_data(attribute_name: str, attribute_definition: str) -> str:
    """Concatenate attribute name and definition for embedding.
    
    Parameters:
        attribute_name (str): Name of the attribute
        attribute_definition (str): Definition of the attribute
        
    Returns:
        str: Concatenated text
    """
    return f"{attribute_name}: {attribute_definition}"

def save_embeddings(embeddings: np.ndarray, output_dir: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Save embeddings to disk with full precision and dimension.
    
    Parameters:
        embeddings (np.ndarray): The embedding vectors to save
        output_dir (Union[str, Path]): Directory to save the embeddings
        metadata (Optional[Dict[str, Any]]): Additional metadata to save with the embeddings
        
    Returns:
        str: Path to the saved embedding file
    """
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        # Always save to an 'embeddings' subfolder
        if output_dir.name != 'embeddings':
            output_dir = output_dir / 'embeddings'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for the filename
        timestamp = int(time.time())
        
        # Create filenames for embeddings and metadata
        embedding_file = output_dir / f"embeddings_{timestamp}.npy"
        metadata_file = output_dir / f"embeddings_{timestamp}_metadata.json"
        
        # Save embeddings using numpy's full precision
        np.save(embedding_file, embeddings)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        # Add embedding information to metadata
        metadata.update({
            "timestamp": timestamp,
            "embedding_file": str(embedding_file),
            "shape": embeddings.shape,
            "dtype": str(embeddings.dtype),
            "dimensions": embeddings.shape[1] if len(embeddings.shape) > 1 else None
        })
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved embeddings to {embedding_file} with shape {embeddings.shape}")
        return str(embedding_file)
        
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        raise

def load_embeddings(embedding_file: Union[str, Path]) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Load embeddings from disk with full precision and dimension.
    
    Parameters:
        embedding_file (Union[str, Path]): Path to the embedding file
        
    Returns:
        Tuple[np.ndarray, Optional[Dict[str, Any]]]: Tuple containing embeddings and metadata
    """
    try:
        embedding_file = Path(embedding_file)
        
        # Check if the file exists
        if not embedding_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        # Load embeddings
        embeddings = np.load(embedding_file)
        
        # Try to load metadata if it exists
        metadata_file = embedding_file.parent / f"{embedding_file.stem}_metadata.json"
        metadata = None
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
        logger.info(f"Loaded embeddings from {embedding_file} with shape {embeddings.shape}")
        return embeddings, metadata
        
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise

def run_embedding(attributes_df, model_path: str, provider: str = "huggingface", 
                 save_to_dir: Optional[Union[str, Path]] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Embed data attributes using the specified model.
    
    Parameters:
        attributes_df: DataFrame containing attribute_name and attribute_definition columns
        model_path (str): Path to the embedding model or name of Azure OpenAI model
        provider (str): Model provider, either 'huggingface' or 'azure_openai'
        save_to_dir (Optional[Union[str, Path]]): Directory to save the embeddings
        metadata (Optional[Dict[str, Any]]): Additional metadata to save with the embeddings
        
    Returns:
        np.ndarray: Array of embedding vectors
    """
    try:
        # Concatenate attribute name and definition
        texts = [concatenate_attribute_data(row['attribute_name'], row['attribute_definition']) 
                for _, row in attributes_df.iterrows()]
        
        # Initialize and load the embedding model
        embedding_model = EmbeddingModel(model_path, provider=provider)
        embedding_model.load_model()
        
        # Get embeddings
        embeddings = embedding_model.get_embeddings(texts)
        
        # Save embeddings if directory is provided
        if save_to_dir is not None:
            # Add model information to metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "model_path": model_path,
                "provider": provider,
                "num_attributes": len(attributes_df),
                "attribute_ids": attributes_df['attribute_id'].tolist() if 'attribute_id' in attributes_df.columns else None
            })
            
            save_embeddings(embeddings, save_to_dir, metadata)
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error in embed_attributes: {e}")
        raise
