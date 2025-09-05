# Generic File Classification System

A machine learning-based file classification system that can predict file types and map columns to templates using fuzzy matching. This is a generic, configurable version that can be easily adapted for different environments and use cases.

## Features

- **File Type Prediction**: Uses TF-IDF vectorization and SVM classification to predict file types based on column names
- **Fuzzy Column Matching**: Maps columns between uploaded files and template files using fuzzy string matching
- **Multi-Encoding Support**: Handles various file encodings (UTF-8, Latin1, CP1252, ANSI)
- **Configurable Storage**: Supports different file storage backends (local filesystem, cloud storage, APIs)
- **Template Mapping**: Flexible template system for different file types
- **Generic Design**: No hardcoded business logic or private information

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd generic-file-classification
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install optional dependencies for your storage backend (see requirements.txt for options)

## Quick Start

```python
from generic_classification import FileClassificationSystem, create_local_file_reader

# Create configuration
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

# Initialize and train the system
classifier = FileClassificationSystem(config)
classifier.train_classifier()

# Process a file
result = classifier.process_file('sample_file.csv')
print(result)
```

## Configuration

The system is highly configurable through a configuration dictionary:

### Required Configuration

- `training_data_path`: Path to your training data CSV file
- `template_paths`: Dictionary mapping file types to template file paths
- `file_reader`: Function to read files from your storage backend

### Optional Configuration

- `fuzzy_match_threshold`: Minimum score for fuzzy matching (default: 90)

## File Reader Examples

### Local Filesystem

```python
def create_local_file_reader(base_path: str = "."):
    def read_file(file_path: str) -> str:
        full_path = os.path.join(base_path, file_path)
        with open(full_path, 'rb') as f:
            return f.read()
    return read_file
```

### Azure Blob Storage

```python
from azure.storage.blob import BlobServiceClient

def create_azure_file_reader(connection_string: str, container_name: str):
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)

    def read_file(file_path: str) -> bytes:
        blob_client = container_client.get_blob_client(file_path)
        return blob_client.download_blob().readall()

    return read_file
```

### AWS S3

```python
import boto3

def create_s3_file_reader(bucket_name: str, aws_access_key: str = None, aws_secret_key: str = None):
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)

    def read_file(file_path: str) -> bytes:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
        return response['Body'].read()

    return read_file
```

## Training Data Format

Your training data should be a CSV file with the following structure:

| column1 | column2 | column3 | ... | file_type  |
| ------- | ------- | ------- | --- | ---------- |
| value1  | value2  | value3  | ... | Materials  |
| value1  | value2  | value3  | ... | Components |
| value1  | value2  | value3  | ... | Suppliers  |

The system uses all columns except `file_type` as features for training.

## Template Files

Template files should be CSV files containing the expected column structure for each file type. The system will map uploaded file columns to these template columns using fuzzy matching.

## API Reference

### FileClassificationSystem

#### `__init__(config: Dict[str, Any])`

Initialize the classification system with configuration.

#### `train_classifier() -> None`

Train the file type classifier using the training data.

#### `process_file(file_path: str) -> Dict[str, Any]`

Process a file and return classification results with column mapping.

Returns:

```python
{
    "importType": "Materials",  # Predicted file type
    "rowCount": 150,           # Number of rows in the file
    "headers": [               # Column mappings
        {
            "initialValue": "Product Name",    # Original column name
            "value": "product_name",          # Mapped template column
            "map": "Product Name"             # Mapping key
        },
        # ... more columns
    ]
}
```

## Error Handling

The system includes comprehensive error handling and logging. All errors are logged with appropriate detail levels:

- `INFO`: Normal operation messages
- `WARNING`: Non-critical issues (e.g., encoding fallbacks)
- `ERROR`: Critical errors that prevent operation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 1.0.0

- Initial release
- Generic file classification system
- Support for multiple storage backends
- Fuzzy column matching
- Multi-encoding support
