"""
Example usage of the Generic File Classification System

This file demonstrates how to use the classification system with different
storage backends and configurations.
"""

import json
import os
from generic_classification import FileClassificationSystem, create_local_file_reader


def example_local_filesystem():
    """Example using local filesystem storage."""
    print("=== Local Filesystem Example ===")
    
    # Configuration for local filesystem
    config = {
        'training_data_path': 'data/training_data.csv',
        'template_paths': {
            'Materials': 'templates/materials_template.csv',
            'Components': 'templates/components_template.csv',
            'Suppliers': 'templates/suppliers_template.csv',
        },
        'file_reader': create_local_file_reader('data'),
        'fuzzy_match_threshold': 90
    }
    
    # Initialize and train
    classifier = FileClassificationSystem(config)
    classifier.train_classifier()
    
    # Process a file
    try:
        result = classifier.process_file('sample_file.csv')
        print("Classification Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")


def example_azure_blob_storage():
    """Example using Azure Blob Storage (requires azure-storage-blob)."""
    print("\n=== Azure Blob Storage Example ===")
    
    # Uncomment and configure for Azure Blob Storage
    """
    from azure.storage.blob import BlobServiceClient
    
    def create_azure_file_reader(connection_string: str, container_name: str):
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)
        
        def read_file(file_path: str) -> bytes:
            blob_client = container_client.get_blob_client(file_path)
            return blob_client.download_blob().readall()
        
        return read_file
    
    # Configuration for Azure Blob Storage
    config = {
        'training_data_path': 'training/bigdata2-csv.csv',
        'template_paths': {
            'Materials': 'templates/Materials Template.csv',
            'Components': 'templates/Components Template.csv',
            'Suppliers': 'templates/Suppliers Template.csv',
        },
        'file_reader': create_azure_file_reader(
            connection_string=os.environ.get('AZURE_STORAGE_CONNECTION_STRING'),
            container_name='your-container'
        ),
        'fuzzy_match_threshold': 90
    }
    
    classifier = FileClassificationSystem(config)
    classifier.train_classifier()
    
    result = classifier.process_file('file-records/sample_file.csv')
    print("Classification Result:")
    print(json.dumps(result, indent=2))
    """
    print("Azure Blob Storage example requires azure-storage-blob package")


def example_aws_s3():
    """Example using AWS S3 (requires boto3)."""
    print("\n=== AWS S3 Example ===")
    
    # Uncomment and configure for AWS S3
    """
    import boto3
    
    def create_s3_file_reader(bucket_name: str, aws_access_key: str = None, aws_secret_key: str = None):
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
        
        def read_file(file_path: str) -> bytes:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            return response['Body'].read()
        
        return read_file
    
    config = {
        'training_data_path': 'training/training_data.csv',
        'template_paths': {
            'Materials': 'templates/materials_template.csv',
            'Components': 'templates/components_template.csv',
            'Suppliers': 'templates/suppliers_template.csv',
        },
        'file_reader': create_s3_file_reader(
            bucket_name='your-bucket-name',
            aws_access_key=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        ),
        'fuzzy_match_threshold': 90
    }
    
    classifier = FileClassificationSystem(config)
    classifier.train_classifier()
    
    result = classifier.process_file('uploads/sample_file.csv')
    print("Classification Result:")
    print(json.dumps(result, indent=2))
    """
    print("AWS S3 example requires boto3 package")


def example_http_api():
    """Example using HTTP API for file access."""
    print("\n=== HTTP API Example ===")
    
    # Uncomment and configure for HTTP API
    """
    import requests
    
    def create_http_file_reader(base_url: str, api_key: str = None):
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        
        def read_file(file_path: str) -> bytes:
            response = requests.get(f"{base_url}/{file_path}", headers=headers)
            response.raise_for_status()
            return response.content
        
        return read_file
    
    config = {
        'training_data_path': 'api/training_data.csv',
        'template_paths': {
            'Materials': 'api/templates/materials_template.csv',
            'Components': 'api/templates/components_template.csv',
            'Suppliers': 'api/templates/suppliers_template.csv',
        },
        'file_reader': create_http_file_reader(
            base_url='https://your-api.com/files',
            api_key=os.environ.get('API_KEY')
        ),
        'fuzzy_match_threshold': 90
    }
    
    classifier = FileClassificationSystem(config)
    classifier.train_classifier()
    
    result = classifier.process_file('uploads/sample_file.csv')
    print("Classification Result:")
    print(json.dumps(result, indent=2))
    """
    print("HTTP API example requires requests package")


def create_sample_training_data():
    """Create sample training data for testing."""
    print("\n=== Creating Sample Training Data ===")
    
    import pandas as pd
    
    # Create sample training data
    training_data = [
        {
            'product_name': 'Steel Rod',
            'material_type': 'Metal',
            'supplier': 'ABC Corp',
            'quantity': 100,
            'file_type': 'Materials'
        },
        {
            'component_id': 'CMP001',
            'component_name': 'Bolt',
            'manufacturer': 'XYZ Inc',
            'specification': 'M8x20',
            'file_type': 'Components'
        },
        {
            'supplier_name': 'DEF Ltd',
            'contact_person': 'John Doe',
            'email': 'john@def.com',
            'phone': '123-456-7890',
            'file_type': 'Suppliers'
        },
        {
            'product_name': 'Aluminum Sheet',
            'material_type': 'Metal',
            'supplier': 'GHI Corp',
            'thickness': '2mm',
            'file_type': 'Materials'
        },
        {
            'component_id': 'CMP002',
            'component_name': 'Washer',
            'manufacturer': 'JKL Inc',
            'specification': 'M8',
            'file_type': 'Components'
        }
    ]
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Save training data
    df = pd.DataFrame(training_data)
    df.to_csv('data/training_data.csv', index=False)
    print("Created sample training data at data/training_data.csv")
    
    # Create sample template files
    materials_template = pd.DataFrame(columns=['product_name', 'material_type', 'supplier', 'quantity'])
    materials_template.to_csv('templates/materials_template.csv', index=False)
    
    components_template = pd.DataFrame(columns=['component_id', 'component_name', 'manufacturer', 'specification'])
    components_template.to_csv('templates/components_template.csv', index=False)
    
    suppliers_template = pd.DataFrame(columns=['supplier_name', 'contact_person', 'email', 'phone'])
    suppliers_template.to_csv('templates/suppliers_template.csv', index=False)
    
    print("Created sample template files in templates/ directory")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_training_data()
    
    # Run examples
    example_local_filesystem()
    example_azure_blob_storage()
    example_aws_s3()
    example_http_api()
