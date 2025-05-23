import re
import spacy
from typing import List, Dict, Union
import numpy as np
from collections import Counter

class QueryPreprocessor:
    def __init__(self, model_name: str = 'en_core_web_sm'):
        """
        Initialize the Query Preprocessor.
        
        Args:
            model_name (str): Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # If model is not found, download it
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model_name])
            self.nlp = spacy.load(model_name)
        
        # Common device-related terms
        self.device_terms = {
            'smartwatch', 'phone', 'laptop', 'printer', 'tablet',
            'desktop', 'monitor', 'keyboard', 'mouse', 'router',
            'modem', 'console', 'gaming', 'device', 'screen'
        }
        
        # Common issue-related terms
        self.issue_terms = {
            'error', 'issue', 'problem', 'bug', 'fault',
            'malfunction', 'broken', 'not working', 'failed',
            'crash', 'frozen', 'stuck', 'slow', 'unresponsive'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, text: str) -> Dict[str, Union[str, List[str], float]]:
        """
        Extract features from query text.
        
        Args:
            text (str): Input query text
            
        Returns:
            dict: Dictionary of extracted features
        """
        # Process text with spaCy
        doc = self.nlp(self.clean_text(text))
        
        # Extract features
        features = {
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'pos_tags': [token.pos_ for token in doc],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks],
            'devices_mentioned': self._extract_device_mentions(doc),
            'issue_indicators': self._extract_issue_indicators(doc),
            'urgency_score': self._calculate_urgency(doc),
            'complexity_score': self._calculate_complexity(doc),
            'key_terms': self._extract_key_terms(doc)
        }
        
        return features
    
    def _extract_device_mentions(self, doc) -> List[str]:
        """Extract mentioned devices from text"""
        devices = []
        for token in doc:
            if token.text.lower() in self.device_terms:
                devices.append(token.text.lower())
        return list(set(devices))
    
    def _extract_issue_indicators(self, doc) -> List[str]:
        """Extract issue-related terms"""
        issues = []
        text = doc.text.lower()
        for issue in self.issue_terms:
            if issue in text:
                issues.append(issue)
        return list(set(issues))
    
    def _calculate_urgency(self, doc) -> float:
        """
        Calculate urgency score based on keywords and patterns.
        Returns score between 0 and 1.
        """
        urgency_indicators = {
            'urgent': 1.0,
            'asap': 1.0,
            'emergency': 1.0,
            'immediately': 0.9,
            'critical': 0.9,
            'urgent': 0.8,
            'soon': 0.6,
            'please': 0.3
        }
        
        score = 0.0
        text = doc.text.lower()
        
        for indicator, weight in urgency_indicators.items():
            if indicator in text:
                score = max(score, weight)
        
        return score
    
    def _calculate_complexity(self, doc) -> float:
        """
        Calculate query complexity score based on various factors.
        Returns score between 0 and 1.
        """
        # Factors contributing to complexity
        num_entities = len(doc.ents)
        num_noun_phrases = len(list(doc.noun_chunks))
        avg_token_length = np.mean([len(token.text) for token in doc])
        num_technical_terms = len(self._extract_device_mentions(doc))
        
        # Combine factors into score
        score = (
            0.3 * min(num_entities / 5, 1) +
            0.2 * min(num_noun_phrases / 8, 1) +
            0.2 * min((avg_token_length - 3) / 5, 1) +
            0.3 * min(num_technical_terms / 3, 1)
        )
        
        return max(0.0, min(1.0, score))
    
    def _extract_key_terms(self, doc) -> List[str]:
        """Extract key terms based on POS tags and dependencies"""
        key_terms = []
        
        # Include nouns and verbs
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop:
                key_terms.append(token.lemma_)
        
        # Include named entities
        for ent in doc.ents:
            key_terms.append(ent.text)
        
        # Count frequencies
        term_freq = Counter(key_terms)
        
        # Return most common terms
        return [term for term, freq in term_freq.most_common(5)]

def main():
    """
    Example usage of the QueryPreprocessor
    """
    # Initialize preprocessor
    preprocessor = QueryPreprocessor()
    
    # Example queries
    queries = [
        "URGENT: My smartwatch is not connecting to the phone and needs immediate attention!",
        "I'm having technical difficulties with my printer's wireless setup.",
        "The gaming console keeps crashing during gameplay, please help."
    ]
    
    # Process queries
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        features = preprocessor.extract_features(query)
        
        print("\nExtracted Features:")
        print(f"Devices mentioned: {features['devices_mentioned']}")
        print(f"Issue indicators: {features['issue_indicators']}")
        print(f"Urgency score: {features['urgency_score']:.2f}")
        print(f"Complexity score: {features['complexity_score']:.2f}")
        print(f"Key terms: {features['key_terms']}")
        print(f"Named entities: {features['entities']}")

if __name__ == "__main__":
    main() 