import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import random

class QueryAugmenter:
    def __init__(self):
        """Initialize the translation models for back-translation"""
        # English to French model
        self.en_fr_model_name = 'Helsinki-NLP/opus-mt-en-fr'
        self.en_fr_tokenizer = MarianTokenizer.from_pretrained(self.en_fr_model_name)
        self.en_fr_model = MarianMTModel.from_pretrained(self.en_fr_model_name)
        
        # French to English model
        self.fr_en_model_name = 'Helsinki-NLP/opus-mt-fr-en'
        self.fr_en_tokenizer = MarianTokenizer.from_pretrained(self.fr_en_model_name)
        self.fr_en_model = MarianMTModel.from_pretrained(self.fr_en_model_name)

    def back_translate(self, text):
        """
        Perform back-translation: English -> French -> English
        
        Args:
            text (str): Original English text
            
        Returns:
            str: Back-translated English text
        """
        # Translate to French
        fr_tokens = self.en_fr_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        fr_translated = self.en_fr_model.generate(**fr_tokens)
        fr_text = self.en_fr_tokenizer.decode(fr_translated[0], skip_special_tokens=True)
        
        # Translate back to English
        en_tokens = self.fr_en_tokenizer(fr_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        en_translated = self.fr_en_model.generate(**en_tokens)
        augmented_text = self.fr_en_tokenizer.decode(en_translated[0], skip_special_tokens=True)
        
        return augmented_text

    def augment_dataset(self, input_file, output_file, augmentation_factor=2):
        """
        Augment the dataset using back-translation
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output CSV file
            augmentation_factor (int): Number of augmented versions to create for each query
        """
        # Read original dataset
        df = pd.read_csv(input_file)
        print(f"Original dataset size: {len(df)} queries")
        
        # Create empty list for augmented data
        augmented_data = []
        
        # Process each query
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting queries"):
            # Add original query
            augmented_data.append(row.to_dict())
            
            # Create augmented versions
            for i in range(augmentation_factor - 1):
                augmented_query = self.back_translate(row['Query_Text'])
                
                # Create new row with augmented query
                new_row = row.to_dict()
                new_row['Query_Text'] = augmented_query
                new_row['Query_ID'] = f"{row['Query_ID']}_aug_{i+1}"
                augmented_data.append(new_row)
        
        # Create augmented dataset
        augmented_df = pd.DataFrame(augmented_data)
        print(f"Augmented dataset size: {len(augmented_df)} queries")
        
        # Save augmented dataset
        augmented_df.to_csv(output_file, index=False)
        print(f"Augmented dataset saved to {output_file}")

def main():
    # Initialize augmenter
    augmenter = QueryAugmenter()
    
    # Define input and output files
    input_file = "Dataset/improved_query_segmentation_dataset.csv"
    output_file = "Dataset/augmented_query_dataset.csv"
    
    # Perform augmentation
    print("Starting data augmentation...")
    augmenter.augment_dataset(input_file, output_file, augmentation_factor=2)
    print("Data augmentation completed!")
    
    # Display sample of augmented queries
    augmented_df = pd.read_csv(output_file)
    print("\nSample of original and augmented queries:")
    sample_queries = augmented_df[augmented_df['Query_ID'].str.contains('Q00001')]['Query_Text'].values
    for i, query in enumerate(sample_queries):
        print(f"\nVersion {i}:")
        print(query)

if __name__ == "__main__":
    main() 