import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer
from typing import Tuple, Dict
import os
from datetime import datetime
import sys
from contextlib import contextmanager

@contextmanager
def capture_output():
    """Capture stdout to a string"""
    from io import StringIO
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

class QueryDataLoader:
    def __init__(self, data_path: str = "Dataset/augmented_query_dataset.csv"):
        """
        Initialize the QueryDataLoader with the augmented dataset.
        
        Args:
            data_path (str): Path to the augmented dataset CSV file
        """
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.department_mapping = None
        self.device_mapping = None
        
    def load_and_preprocess(self):
        """
        Load and preprocess the augmented dataset.
        """
        # Read the CSV file
        self.df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime and extract time features
        self.df['Time_of_Day'] = pd.to_datetime(self.df['Time_of_Day'])
        self.df['Hour'] = self.df['Time_of_Day'].dt.hour
        self.df['Day_of_Week'] = self.df['Time_of_Day'].dt.dayofweek
        
        # Encode departments and devices
        self.df['Department_Code'] = self.label_encoder.fit_transform(self.df['Department'])
        self.department_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_)
        ))
        
        # Create device encoding
        unique_devices = self.df['Device'].unique()
        self.device_mapping = {device: idx for idx, device in enumerate(unique_devices)}
        self.df['Device_Code'] = self.df['Device'].map(self.device_mapping)
        
        return self.df
    
    def prepare_train_val_test_split(self, 
                                   val_size: float = 0.1,
                                   test_size: float = 0.1,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into training, validation, and test sets.
        Ensures that augmented versions of queries stay in the same split as their originals.
        
        Args:
            val_size (float): Proportion of data for validation
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Get unique base query IDs (without _aug_ suffix)
        base_query_ids = self.df['Query_ID'].apply(lambda x: x.split('_aug_')[0]).unique()
        
        # First split: train + val vs test
        train_val_ids, test_ids = train_test_split(
            base_query_ids,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: train vs val
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_size/(1-test_size),
            random_state=random_state
        )
        
        # Create masks for the full dataset including augmented queries
        train_mask = self.df['Query_ID'].apply(lambda x: x.split('_aug_')[0] in train_ids)
        val_mask = self.df['Query_ID'].apply(lambda x: x.split('_aug_')[0] in val_ids)
        test_mask = self.df['Query_ID'].apply(lambda x: x.split('_aug_')[0] in test_ids)
        
        return self.df[train_mask], self.df[val_mask], self.df[test_mask]

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save the data splits to separate CSV files.
        """
        # Create splits directory if it doesn't exist
        os.makedirs("Dataset/splits", exist_ok=True)
        
        # Save splits
        train_df.to_csv("Dataset/splits/train.csv", index=False)
        val_df.to_csv("Dataset/splits/validation.csv", index=False)
        test_df.to_csv("Dataset/splits/test.csv", index=False)
    
    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get the dimensions of various features in the dataset.
        
        Returns:
            dict: Dictionary containing feature dimensions
        """
        return {
            'n_departments': len(self.department_mapping),
            'n_devices': len(self.device_mapping),
            'max_sequence_length': 512  # BERT's max sequence length
        }
    
    def tokenize_text(self, texts: list) -> dict:
        """
        Tokenize a list of texts using BERT tokenizer.
        
        Args:
            texts (list): List of text strings to tokenize
            
        Returns:
            dict: Dictionary of tokenized inputs ready for BERT
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

def save_statistics_report(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                         test_df: pd.DataFrame, feature_dims: Dict):
    """Save detailed statistics to a report file"""
    # Create reports directory if it doesn't exist
    os.makedirs("Dataset/reports", exist_ok=True)
    
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"Dataset/reports/dataset_statistics_{timestamp}.txt"
    
    # Capture all statistics output
    with capture_output() as output:
        # Print full dataset statistics
        print_dataset_statistics(df, "Full")
        
        # Print split statistics
        print_dataset_statistics(train_df, "Training")
        print_dataset_statistics(val_df, "Validation")
        print_dataset_statistics(test_df, "Test")
        
        # Print feature dimensions
        print("\nFeature Dimensions:")
        for key, value in feature_dims.items():
            print(f"{key}: {value}")
        
        # Print split ratios
        total = len(df)
        print("\nSplit Ratios:")
        print(f"Training: {len(train_df)/total*100:.1f}%")
        print(f"Validation: {len(val_df)/total*100:.1f}%")
        print(f"Test: {len(test_df)/total*100:.1f}%")
        
        # Print additional metadata
        print("\nDataset Metadata:")
        print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total number of unique departments: {len(df['Department'].unique())}")
        print(f"Total number of unique devices: {len(df['Device'].unique())}")
        print(f"Date range: {df['Time_of_Day'].min()} to {df['Time_of_Day'].max()}")
    
    # Save the captured output to file
    with open(report_file, 'w') as f:
        f.write(output.getvalue())
    
    return report_file

def print_dataset_statistics(df: pd.DataFrame, split_name: str = ""):
    """Print detailed statistics about the dataset"""
    print(f"\n{'='*20} {split_name} Dataset Statistics {'='*20}")
    print(f"Number of samples: {len(df)}")
    
    # Original vs Augmented queries
    n_original = df[~df['Query_ID'].str.contains('_aug_')].shape[0]
    n_augmented = df[df['Query_ID'].str.contains('_aug_')].shape[0]
    print(f"\nQuery Distribution:")
    print(f"Original queries: {n_original}")
    print(f"Augmented queries: {n_augmented}")
    
    # Department distribution
    print(f"\nDepartment Distribution:")
    dept_dist = df['Department'].value_counts()
    for dept, count in dept_dist.items():
        print(f"{dept}: {count} ({count/len(df)*100:.1f}%)")
    
    # Device distribution
    print(f"\nDevice Distribution:")
    device_dist = df['Device'].value_counts()
    for device, count in device_dist.items():
        print(f"{device}: {count} ({count/len(df)*100:.1f}%)")
    
    # Time distribution
    print(f"\nTime Distribution:")
    print("Hour of day distribution:")
    hour_dist = df['Hour'].value_counts().sort_index()
    for hour, count in hour_dist.items():
        print(f"{hour:02d}:00 - {hour:02d}:59: {count} queries")
    
    print("\nDay of week distribution:")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_dist = df['Day_of_Week'].value_counts().sort_index()
    for day_num, count in day_dist.items():
        print(f"{days[day_num]}: {count} queries")

def main():
    """
    Example usage of the QueryDataLoader with detailed statistics
    """
    # Initialize the data loader
    loader = QueryDataLoader()
    
    # Load and preprocess the data
    print("\nLoading and preprocessing the dataset...")
    df = loader.load_and_preprocess()
    
    # Split the data
    print("\nSplitting dataset into train/validation/test sets...")
    train_df, val_df, test_df = loader.prepare_train_val_test_split()
    
    # Save splits
    loader.save_splits(train_df, val_df, test_df)
    print("\nSaved dataset splits to Dataset/splits/")
    
    # Get feature dimensions
    dims = loader.get_feature_dims()
    
    # Save statistics report
    report_file = save_statistics_report(df, train_df, val_df, test_df, dims)
    print(f"\nDetailed statistics saved to: {report_file}")
    
    # Print statistics to console
    print_dataset_statistics(df, "Full")
    print_dataset_statistics(train_df, "Training")
    print_dataset_statistics(val_df, "Validation")
    print_dataset_statistics(test_df, "Test")
    
    # Print feature dimensions
    print("\nFeature Dimensions:")
    for key, value in dims.items():
        print(f"{key}: {value}")
    
    # Example of tokenization
    print("\nTokenization Example:")
    sample_texts = train_df['Query_Text'].head(2).tolist()
    tokens = loader.tokenize_text(sample_texts)
    print("Sample text:")
    print(sample_texts[0])
    print(f"Tokenized shape: {tokens['input_ids'].shape}")
    print(f"Number of tokens: {len(tokens['input_ids'][0])}")

if __name__ == "__main__":
    main() 