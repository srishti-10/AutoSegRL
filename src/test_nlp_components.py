from models.bert_encoder import QueryEncoderWithDevice
from models.text_processor import QueryPreprocessor
from models.query_store import QueryStore
import torch
import uuid

def test_components():
    print("Initializing NLP Components...")
    
    # Initialize components
    encoder = QueryEncoderWithDevice(model_name='bert-base-uncased')
    preprocessor = QueryPreprocessor()
    store = QueryStore()
    
    # Test queries
    test_queries = [
        "URGENT: My laptop keeps crashing during video calls!",
        "How do I connect my smartwatch to my phone?",
        "The printer is showing a paper jam error, but there's no paper stuck.",
        "Need immediate help with gaming console not turning on"
    ]
    
    print("\nProcessing Test Queries...")
    print("-" * 50)
    
    stored_ids = []  # Keep track of stored query IDs
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        query_id = str(uuid.uuid4())
        stored_ids.append(query_id)
        
        # Get text features
        features = preprocessor.extract_features(query)
        
        # Get BERT embeddings
        embeddings, attention_mask = encoder.encode(query)
        
        # Store query
        store.store_query(
            query_id=query_id,
            query_text=query,
            embeddings=embeddings,
            features=features,
            attention_mask=attention_mask
        )
        
        # Print results
        print("\nText Analysis:")
        print(f"- Devices mentioned: {features['devices_mentioned']}")
        print(f"- Issue indicators: {features['issue_indicators']}")
        print(f"- Urgency score: {features['urgency_score']:.2f}")
        print(f"- Complexity score: {features['complexity_score']:.2f}")
        print(f"- Key terms: {features['key_terms']}")
        
        print("\nBERT Encoding:")
        print(f"- Embedding shape: {embeddings.shape}")
        print(f"- Embedding norm: {torch.norm(embeddings).item():.2f}")
        
        # If not the last query, calculate similarity with next query
        if i < len(test_queries):
            next_embedding, _ = encoder.encode(test_queries[i])
            similarity = torch.nn.functional.cosine_similarity(embeddings, next_embedding)
            print(f"\nSimilarity with next query: {similarity.item():.4f}")
        
        print("-" * 50)
    
    # Demonstrate loading stored queries
    print("\nDemonstrating Query Storage and Retrieval:")
    print("-" * 50)
    
    for query_id in stored_ids:
        text, embeddings, features = store.load_query(query_id)
        print(f"\nLoaded Query ID: {query_id}")
        print(f"Text: {text}")
        print(f"Features: {features['key_terms']}")
        print(f"Embedding shape: {embeddings.shape}")
    
    # Demonstrate similarity search
    print("\nDemonstrating Similarity Search:")
    print("-" * 50)
    
    # Use the first query as an example
    example_text, example_embeddings, _ = store.load_query(stored_ids[0])
    similar_queries = store.search_similar_queries(example_embeddings, top_k=3)
    
    print(f"\nFinding similar queries to: {example_text}")
    for query_id, similarity in similar_queries:
        text, _, _ = store.load_query(query_id)
        print(f"- {text} (similarity: {similarity:.4f})")

if __name__ == "__main__":
    test_components() 