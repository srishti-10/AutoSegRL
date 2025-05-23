import torch
import json
import os
from typing import Dict, List, Union, Tuple
import numpy as np
from datetime import datetime

class QueryStore:
    def __init__(self, store_dir: str = "data/processed_queries"):
        """
        Initialize QueryStore with a storage directory.
        
        Args:
            store_dir (str): Directory to store processed queries
        """
        self.store_dir = store_dir
        self.embeddings_dir = os.path.join(store_dir, "embeddings")
        self.features_dir = os.path.join(store_dir, "features")
        self.metadata_file = os.path.join(store_dir, "metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Initialize or load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file or create new if doesn't exist"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "queries": {},
            "total_count": 0,
            "last_updated": None
        }
    
    def _save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def store_query(self, 
                    query_id: str,
                    query_text: str,
                    embeddings: torch.Tensor,
                    features: Dict,
                    attention_mask: torch.Tensor = None) -> str:
        """
        Store a processed query with its embeddings and features.
        
        Args:
            query_id (str): Unique identifier for the query
            query_text (str): Original query text
            embeddings (torch.Tensor): BERT embeddings
            features (dict): Extracted features from text processor
            attention_mask (torch.Tensor, optional): Attention mask from BERT
            
        Returns:
            str: Query ID
        """
        # Convert embeddings to numpy and save
        embeddings_np = embeddings.cpu().numpy()
        embedding_file = os.path.join(self.embeddings_dir, f"{query_id}.npy")
        np.save(embedding_file, embeddings_np)
        
        # Save attention mask if provided
        if attention_mask is not None:
            mask_file = os.path.join(self.embeddings_dir, f"{query_id}_mask.npy")
            np.save(mask_file, attention_mask.cpu().numpy())
        
        # Save features
        features_file = os.path.join(self.features_dir, f"{query_id}.json")
        with open(features_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        # Update metadata
        self.metadata["queries"][query_id] = {
            "text": query_text,
            "timestamp": datetime.now().isoformat(),
            "embedding_file": embedding_file,
            "features_file": features_file
        }
        self.metadata["total_count"] += 1
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        self._save_metadata()
        return query_id
    
    def load_query(self, query_id: str) -> Tuple[str, torch.Tensor, Dict]:
        """
        Load a stored query by ID.
        
        Args:
            query_id (str): Query identifier
            
        Returns:
            tuple: (query_text, embeddings, features)
        """
        if query_id not in self.metadata["queries"]:
            raise KeyError(f"Query ID {query_id} not found")
        
        query_data = self.metadata["queries"][query_id]
        
        # Load embeddings
        embeddings = torch.from_numpy(
            np.load(query_data["embedding_file"])
        )
        
        # Load features
        with open(query_data["features_file"], 'r') as f:
            features = json.load(f)
        
        return query_data["text"], embeddings, features
    
    def get_all_query_ids(self) -> List[str]:
        """Get list of all stored query IDs"""
        return list(self.metadata["queries"].keys())
    
    def get_query_count(self) -> int:
        """Get total number of stored queries"""
        return self.metadata["total_count"]
    
    def search_similar_queries(self, 
                             embeddings: torch.Tensor,
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar queries based on embedding similarity.
        
        Args:
            embeddings (torch.Tensor): Query embeddings to compare against
            top_k (int): Number of similar queries to return
            
        Returns:
            list: List of (query_id, similarity_score) tuples
        """
        similarities = []
        query_embeddings = embeddings.cpu()
        
        for query_id in self.get_all_query_ids():
            stored_text, stored_embeddings, _ = self.load_query(query_id)
            similarity = torch.nn.functional.cosine_similarity(
                query_embeddings,
                stored_embeddings,
                dim=1
            ).item()
            similarities.append((query_id, similarity))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def main():
    """Example usage of QueryStore"""
    from bert_encoder import QueryEncoderWithDevice
    from text_processor import QueryPreprocessor
    import uuid
    
    # Initialize components
    store = QueryStore()
    encoder = QueryEncoderWithDevice()
    preprocessor = QueryPreprocessor()
    
    # Example query
    query_text = "My laptop keeps crashing during video calls"
    query_id = str(uuid.uuid4())
    
    # Process query
    features = preprocessor.extract_features(query_text)
    embeddings, attention_mask = encoder.encode(query_text)
    
    # Store query
    store.store_query(
        query_id=query_id,
        query_text=query_text,
        embeddings=embeddings,
        features=features,
        attention_mask=attention_mask
    )
    
    # Load and verify
    loaded_text, loaded_embeddings, loaded_features = store.store_query(query_id)
    print(f"Stored queries: {store.get_query_count()}")
    print(f"Original text: {query_text}")
    print(f"Loaded text: {loaded_text}")
    print(f"Embeddings match: {torch.allclose(embeddings, loaded_embeddings)}")

if __name__ == "__main__":
    main() 