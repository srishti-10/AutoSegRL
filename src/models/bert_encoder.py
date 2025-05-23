import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union, Tuple

class QueryEncoder(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        """
        Initialize the Query Encoder with a pre-trained BERT model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            max_length (int): Maximum sequence length for tokenization
        """
        super(QueryEncoder, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters (optional, can be fine-tuned if needed)
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize input texts.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Dictionary containing tokenized inputs
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def encode(self, texts: Union[str, List[str]], 
               pooling: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts into vector representations.
        
        Args:
            texts: Text or list of texts to encode
            pooling: Pooling strategy ('mean' or 'cls')
            
        Returns:
            tuple: (encoded_vectors, attention_masks)
        """
        # Tokenize inputs
        encoded = self.tokenize(texts)
        
        # Get BERT outputs
        with torch.no_grad():
            outputs = self.bert(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                return_dict=True
            )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling
        if pooling == 'cls':
            # Use [CLS] token representation
            pooled = hidden_states[:, 0, :]
        else:
            # Mean pooling
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        
        return pooled, encoded['attention_mask']
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the output embeddings.
        
        Returns:
            int: Embedding dimension
        """
        return self.bert.config.hidden_size

class QueryEncoderWithDevice(QueryEncoder):
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Query Encoder with specific device support.
        
        Args:
            model_name (str): Name of the pre-trained model
            max_length (int): Maximum sequence length
            device (str): Device to use ('cuda' or 'cpu')
        """
        super(QueryEncoderWithDevice, self).__init__(model_name, max_length)
        self.device = device
        self.to(device)
    
    def encode(self, texts: Union[str, List[str]], 
               pooling: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts with device support.
        """
        encoded = self.tokenize(texts)
        
        # Move inputs to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.bert(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                return_dict=True
            )
        
        hidden_states = outputs.last_hidden_state
        
        if pooling == 'cls':
            pooled = hidden_states[:, 0, :]
        else:
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = torch.sum(hidden_states * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        
        return pooled, encoded['attention_mask']

def main():
    """
    Example usage of the QueryEncoder
    """
    # Initialize encoder
    encoder = QueryEncoderWithDevice()
    
    # Example queries
    queries = [
        "I'm having issues with my smartwatch not connecting to my phone.",
        "Need help with setting up my new gaming console.",
        "My printer is showing error code 501."
    ]
    
    # Encode queries
    embeddings, masks = encoder.encode(queries)
    
    # Print results
    print(f"Encoder model: {encoder.model_name}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    print(f"Number of queries: {len(queries)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Attention masks shape: {masks.shape}")
    
    # Example of similarity calculation
    from torch.nn.functional import cosine_similarity
    sim = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    print(f"\nSimilarity between first two queries: {sim.item():.4f}")

if __name__ == "__main__":
    main() 