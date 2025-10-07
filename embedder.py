from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import torch
import logging
import numpy as np
from config import CACHE_DIR  

class Embedder(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", device=None, 
                 batch_size=32, show_progress_default=False, 
                 cache_embeddings=True):
        try:
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                    logging.info("Using CUDA for embeddings")
                else:
                    device = "cpu"
                    logging.info("Using CPU for embeddings")
            
            self.device = device
            self.model_name = model_name
            self.batch_size = batch_size
            self.show_progress_default = show_progress_default
            self.cache_embeddings = cache_embeddings
            cache_dir = f"{CACHE_DIR}/models" if cache_embeddings else None
            self.model = SentenceTransformer(
                model_name, 
                device=device,
                cache_folder=cache_dir
            )
            self.model.max_seq_length = self.model.get_max_seq_length()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logging.info(f"Loaded embedding model: {model_name}")
            logging.info(f"Embedding dimension: {self.embedding_dim}")
            logging.info(f"Max sequence length: {self.model.max_seq_length}")
            
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model {model_name}: {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}")

    def warmup(self, sample_texts=None):
        if sample_texts is None:
            sample_texts = [
                "Machine learning transformer models",
                "Natural language processing research",
                "Computer vision deep learning",
                "Artificial intelligence algorithms"
            ]
        
        try:
            embeddings = self.encode(sample_texts, show_progress=False, batch_size=2)
            import time
            start_time = time.time()
            _ = self.encode(sample_texts * 10, show_progress=False, batch_size=8)
            warmup_time = time.time() - start_time
            
            logging.info(f"Embedder warmed up. Benchmark: {warmup_time:.2f}s for 40 texts")
            return True
            
        except Exception as e:
            logging.warning(f"Warmup failed: {str(e)}")
            return False

    def _validate_input(self, texts):
        if not isinstance(texts, (list, tuple)):
            raise ValueError(f"Input must be list or tuple, got {type(texts)}")
        
        if not texts:
            raise ValueError("Input list cannot be empty")
        valid_texts = []
        invalid_indices = []
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text)
            else:
                invalid_indices.append(i)
        
        if invalid_indices:
            logging.warning(f"Filtered out {len(invalid_indices)} invalid texts at indices {invalid_indices}")
        
        if not valid_texts:
            raise ValueError("No valid text inputs after filtering")
            
        return valid_texts

    def _check_sequence_length(self, texts):
        max_len = self.model.max_seq_length
        long_texts = []
        
        for i, text in enumerate(texts):
            tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_len:
                long_texts.append((i, len(tokens)))
        
        if long_texts:
            avg_excess = sum(excess for _, excess in long_texts) / len(long_texts)
            logging.warning(
                f"{len(long_texts)} texts exceed max sequence length ({max_len}). "
                f"Average excess: {avg_excess:.0f} tokens. They will be truncated."
            )
        
        return len(long_texts)

    def encode(self, texts, show_progress=None, return_type="list", 
               normalize_embeddings=True, batch_size=None):
        if show_progress is None:
            show_progress = self.show_progress_default
            
        if batch_size is None:
            batch_size = self.batch_size

        try:
            valid_texts = self._validate_input(texts)
            long_count = self._check_sequence_length(valid_texts)
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_tensor=(return_type == "tensor"),
                show_progress_bar=show_progress,
                normalize_embeddings=normalize_embeddings,
                precision='float32'  
            )
            
            if return_type == "numpy":
                if isinstance(embeddings, torch.Tensor):
                    return embeddings.cpu().numpy()
                return embeddings
            elif return_type == "tensor":
                if not isinstance(embeddings, torch.Tensor):
                    return torch.tensor(embeddings)
                return embeddings
            else:
                if isinstance(embeddings, torch.Tensor):
                    return embeddings.cpu().tolist()
                elif isinstance(embeddings, np.ndarray):
                    return embeddings.tolist()
                return embeddings
                
        except Exception as e:
            logging.error(f"Text encoding failed: {str(e)}")
            raise RuntimeError(f"Failed to encode {len(texts)} texts: {str(e)}")

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.max_seq_length,
            "device": str(self.device),
            "batch_size": self.batch_size
        }

    def __call__(self, input):
        return self.encode(input, show_progress=self.show_progress_default, return_type="list")

# Utility function for embedding comparison
def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
    
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Global embedder instance for reuse
_global_embedder = None

def get_global_embedder(**kwargs):
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = Embedder(**kwargs)
        _global_embedder.warmup()
    return _global_embedder