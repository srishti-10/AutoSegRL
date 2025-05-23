# Query Segregation System

An intelligent system for automatically routing customer queries to appropriate departments using NLP and machine learning techniques.

## Project Overview

The Query Segregation System uses state-of-the-art NLP techniques to analyze, process, and route customer queries efficiently. It combines BERT-based embeddings with feature extraction to understand query context, urgency, and complexity.

## Components

### 1. NLP Pipeline
- **BERT Query Encoder** (`src/models/bert_encoder.py`)
  - Uses BERT for semantic query encoding
  - Supports both CPU and GPU processing
  - Configurable pooling strategies (mean and CLS token)
  - Handles batch processing efficiently

- **Text Processor** (`src/models/text_processor.py`)
  - Extracts key features from queries:
    - Device mentions
    - Issue indicators
    - Urgency scoring
    - Complexity scoring
    - Key terms extraction
  - Uses spaCy for efficient NLP operations

- **Query Store** (`src/models/query_store.py`)
  - Persistent storage for processed queries
  - Stores embeddings and features
  - Supports similarity search
  - Maintains query metadata

### 2. Data Management
- **Dataset Structure**:
  ```
  data/
  ├── processed_queries/
  │   ├── embeddings/     # BERT embeddings (.npy)
  │   ├── features/       # Extracted features (.json)
  │   └── metadata.json   # Query tracking and metadata
  └── splits/
      ├── train/
      ├── validation/
      └── test/
  ```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd query-segregation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### 1. Processing Queries
```python
from models.bert_encoder import QueryEncoderWithDevice
from models.text_processor import QueryPreprocessor
from models.query_store import QueryStore
import uuid

# Initialize components
encoder = QueryEncoderWithDevice()
preprocessor = QueryPreprocessor()
store = QueryStore()

# Process a query
query_text = "My laptop keeps crashing during video calls"
query_id = str(uuid.uuid4())

# Extract features and embeddings
features = preprocessor.extract_features(query_text)
embeddings, attention_mask = encoder.encode(query_text)

# Store processed query
store.store_query(
    query_id=query_id,
    query_text=query_text,
    embeddings=embeddings,
    features=features,
    attention_mask=attention_mask
)
```

### 2. Finding Similar Queries
```python
# Load a query
text, embeddings, features = store.load_query(query_id)

# Find similar queries
similar_queries = store.search_similar_queries(embeddings, top_k=5)
```

## Features

### 1. Query Analysis
- Semantic understanding using BERT
- Feature extraction:
  - Device identification
  - Issue detection
  - Urgency assessment
  - Complexity scoring
  - Key term extraction

### 2. Storage and Retrieval
- Persistent storage of processed queries
- Efficient embedding storage using NumPy
- JSON-based feature storage
- Metadata tracking
- Similarity-based query search

### 3. Performance
- GPU support for BERT encoding
- Efficient batch processing
- Optimized storage format
- Fast similarity search

## Project Status

### Completed
- ✅ Dataset augmentation and preprocessing
- ✅ BERT-based query encoding
- ✅ Feature extraction pipeline
- ✅ Query storage system
- ✅ Similarity search functionality

### In Progress
- 🔄 Query classification model
- 🔄 Contextual bandit implementation
- 🔄 API development
- 🔄 Feedback system

### Planned
- ⏳ Training pipeline
- ⏳ Model evaluation metrics
- ⏳ Streamlit demo application
- ⏳ Production deployment

## Dependencies

- Python 3.8+
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- spaCy >= 3.5.0
- NumPy >= 1.24.0
- scikit-learn >= 1.2.0

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 