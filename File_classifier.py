"""
Generic File Classification System

A machine learning-based file classification system that can predict file types
and map columns to templates using fuzzy matching. This is a generic version
that can be easily configured for different environments and use cases.

Features:
- File type prediction using TF-IDF vectorization and SVM classification
- Fuzzy column matching between uploaded files and template files
- Support for multiple file encodings
- Configurable template mapping
- Generic file handling (local filesystem or cloud storage)

Author: [Your Name]
Version: 1.0.0
"""

import logging
import pandas as pd
import json
import io
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Machine learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from fuzzywuzzy import process


class FileClassificationSystem:
    """
    A generic file classification system that can be configured for different
    environments and file storage backends.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classification system with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - training_data_path: Path to training data CSV
                - template_paths: Dictionary mapping file types to template paths
                - file_reader: Function to read files (local, cloud, etc.)
                - fuzzy_match_threshold: Minimum score for fuzzy matching (default: 90)
        """
        self.config = config
        self.vectorizer = None
        self.classifier = None
        self.training_data = None
        self.template_data = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training data for the classifier.
        
        Returns:
            Tuple of (features, labels)
        """
        try:
            training_path = self.config['training_data_path']
            self.logger.info(f"Loading training data from: {training_path}")
            
            # Use configured file reader
            file_content = self.config['file_reader'](training_path)
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            train_data = pd.read_csv(io.StringIO(file_content))
            X = train_data.drop("file_type", axis=1)  # Features
            y = train_data["file_type"]  # Labels
            
            self.training_data = train_data
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def train_classifier(self) -> None:
        """
        Train the file type classifier using TF-IDF vectorization and SVM.
        """
        try:
            # Load training data
            X_train, y_train = self.load_training_data()
            
            # Convert column names into text for vectorization
            X_train_text = X_train.apply(lambda row: " ".join(map(str, row)), axis=1)
            
            # Vectorize the column names
            self.vectorizer = TfidfVectorizer()
            X_train_vectorized = self.vectorizer.fit_transform(X_train_text)
            
            # Train the classifier
            self.classifier = SVC(kernel="linear")
            self.classifier.fit(X_train_vectorized, y_train)
            
            self.logger.info("Classifier training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training classifier: {str(e)}")
            raise
    
    def predict_file_type(self, file_content: str) -> str:
        """
        Predict the file type based on column names.
        
        Args:
            file_content: Content of the file to classify
            
        Returns:
            Predicted file type
        """
        if not self.classifier or not self.vectorizer:
            raise ValueError("Classifier not trained. Call train_classifier() first.")
        
        try:
            # Vectorize the file content
            file_vectorized = self.vectorizer.transform([file_content])
            predicted_type = self.classifier.predict(file_vectorized)
            return predicted_type[0]
            
        except Exception as e:
            self.logger.error(f"Error predicting file type: {str(e)}")
            raise
    
    def load_file_with_encoding(self, file_path: str) -> Tuple[Optional[pd.DataFrame], int]:
        """
        Load a file with multiple encoding attempts.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Tuple of (DataFrame, row_count) or (None, 0) if failed
        """
        encodings_to_try = ["utf-8", "latin1", "cp1252", "ANSI"]
        
        for encoding in encodings_to_try:
            try:
                file_content = self.config['file_reader'](file_path)
                if isinstance(file_content, bytes):
                    file_content = file_content.decode(encoding)
                
                df = pd.read_csv(io.StringIO(file_content), encoding=encoding)
                row_count = len(df)
                self.logger.info(f"Successfully loaded file with {encoding} encoding")
                return df, row_count
                
            except (UnicodeDecodeError, Exception) as e:
                self.logger.warning(f"Failed to load with {encoding} encoding: {e}")
                continue
        
        self.logger.error("Unable to load file with any of the specified encodings")
        return None, 0
    
    def find_best_column_match(self, column: str, template_columns: List[str]) -> Optional[str]:
        """
        Find the best matching column using fuzzy matching.
        
        Args:
            column: Column name to match
            template_columns: List of template column names
            
        Returns:
            Best matching column name or None if no good match
        """
        threshold = self.config.get('fuzzy_match_threshold', 90)
        best_match, score = process.extractOne(column, template_columns)
        
        if score >= threshold:
            return best_match
        return None
    
    def create_column_mapping(self, predicted_columns: List[str], template_columns: List[str]) -> Dict[str, str]:
        """
        Create a mapping between predicted file columns and template columns.
        
        Args:
            predicted_columns: Columns from the uploaded file
            template_columns: Columns from the template file
            
        Returns:
            Dictionary mapping predicted columns to template columns
        """
        column_mapping = {}
        mapped_template_columns = set()
        
        for column in predicted_columns:
            matched_column = self.find_best_column_match(column, template_columns)
            if matched_column and matched_column not in mapped_template_columns:
                column_mapping[column] = matched_column
                mapped_template_columns.add(matched_column)
        
        return column_mapping
    
    def load_template_file(self, file_type: str) -> Optional[pd.DataFrame]:
        """
        Load the template file for the given file type.
        
        Args:
            file_type: Type of file to get template for
            
        Returns:
            Template DataFrame or None if not found
        """
        if file_type not in self.config['template_paths']:
            self.logger.error(f"No template found for file type: {file_type}")
            return None
        
        template_path = self.config['template_paths'][file_type]
        
        try:
            file_content = self.config['file_reader'](template_path)
            if isinstance(file_content, bytes):
                file_content = file_content.decode('utf-8')
            
            template_df = pd.read_csv(io.StringIO(file_content))
            self.template_data[file_type] = template_df
            return template_df
            
        except Exception as e:
            self.logger.error(f"Error loading template file {template_path}: {e}")
            return None
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a file and return classification results with column mapping.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing:
                - importType: Predicted file type
                - rowCount: Number of rows in the file
                - headers: List of column mappings
        """
        try:
            # Load the file
            predicted_file, row_count = self.load_file_with_encoding(file_path)
            if predicted_file is None:
                raise ValueError("Unable to load the file")
            
            # Prepare file content for prediction (column names as text)
            file_content = " ".join(predicted_file.columns.astype(str))
            
            # Predict file type
            predicted_file_type = self.predict_file_type(file_content)
            self.logger.info(f"Predicted file type: {predicted_file_type}")
            
            # Load template file
            template_data = self.load_template_file(predicted_file_type)
            if template_data is None:
                raise ValueError(f"Unable to load template for file type: {predicted_file_type}")
            
            # Create column mapping
            column_mapping = self.create_column_mapping(
                predicted_file.columns.tolist(),
                template_data.columns.tolist()
            )
            
            # Prepare output
            output = {
                "importType": predicted_file_type,
                "rowCount": row_count,
                "headers": []
            }
            
            # Populate headers with mappings
            for column in predicted_file.columns:
                mapped_column = column_mapping.get(column)
                output["headers"].append({
                    "initialValue": column,
                    "value": mapped_column,
                    "map": column
                })
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            raise


# Example configuration and usage
def create_local_file_reader(base_path: str = "."):
    """
    Create a file reader function for local filesystem.
    
    Args:
        base_path: Base directory for file paths
        
    Returns:
        Function that reads files from local filesystem
    """
    def read_file(file_path: str) -> str:
        full_path = os.path.join(base_path, file_path)
        with open(full_path, 'rb') as f:
            return f.read()
    return read_file


def create_example_config() -> Dict[str, Any]:
    """
    Create an example configuration for the classification system.
    
    Returns:
        Example configuration dictionary
    """
    return {
        'training_data_path': 'data/training_data.csv',
        'template_paths': {
            'Materials': 'templates/materials_template.csv',
            'Components': 'templates/components_template.csv',
            'Suppliers': 'templates/suppliers_template.csv',
        },
        'file_reader': create_local_file_reader('data'),
        'fuzzy_match_threshold': 90
    }


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_example_config()
    
    # Initialize classification system
    classifier = FileClassificationSystem(config)
    
    # Train the classifier
    classifier.train_classifier()
    
    # Process a file
    try:
        result = classifier.process_file('sample_file.csv')
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
