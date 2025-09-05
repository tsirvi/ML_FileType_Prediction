"""
Setup script for Generic File Classification System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="generic-file-classification",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    description="A machine learning-based file classification system with fuzzy column matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[your-username]/generic-file-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "azure": ["azure-storage-blob>=12.14.0", "azure-identity>=1.12.0", "azure-keyvault-secrets>=4.6.0"],
        "aws": ["boto3>=1.26.0"],
        "gcp": ["google-cloud-storage>=2.7.0"],
        "http": ["requests>=2.28.0"],
        "excel": ["openpyxl>=3.0.10", "xlrd>=2.0.1"],
    },
    entry_points={
        "console_scripts": [
            "file-classifier=generic_classification:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
