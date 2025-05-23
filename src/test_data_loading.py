from preprocessing.text_processor import QueryDataProcessor
import pandas as pd

def main():
    # Initialize the data processor
    processor = QueryDataProcessor("Dataset/improved_query_segmentation_dataset.csv")
    
    # Load and preprocess the data
    df = processor.load_data()
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Total number of queries: {len(df)}")
    print("\nDepartment distribution:")
    print(df['Department'].value_counts())
    
    print("\nDevice distribution:")
    print(df['Device'].value_counts())
    
    # Get department mapping
    dept_mapping = processor.get_department_mapping()
    print("\nDepartment Encoding Mapping:")
    for dept, code in dept_mapping.items():
        print(f"{dept}: {code}")
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split()
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

if __name__ == "__main__":
    main() 