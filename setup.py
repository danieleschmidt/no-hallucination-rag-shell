#!/usr/bin/env python
"""
Setup configuration for Quantum-Enhanced RAG System
Production-ready package with quantum-inspired task planning
"""

import os
from setuptools import setup, find_packages

# Read long description from README
def read_file(filename):
    """Read file contents."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        return f.read()

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.21.0,<2.0.0",
    "networkx>=2.6,<4.0.0", 
    "rich>=12.0.0,<14.0.0",
    "pydantic>=1.10.0,<3.0.0",
    "pyyaml>=6.0,<7.0",
    "python-dotenv>=0.19.0,<2.0.0",
    "click>=8.0.0,<9.0.0",
    "cryptography>=3.4.8,<42.0.0",
    "prometheus-client>=0.15.0,<1.0.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.0.0,<8.0.0",
    "pytest-cov>=4.0.0,<5.0.0",
    "pytest-asyncio>=0.20.0,<1.0.0",
    "black>=22.0.0,<24.0.0",
    "flake8>=5.0.0,<7.0.0",
    "mypy>=0.991,<2.0.0",
    "bandit>=1.7.0,<2.0.0",
    "safety>=2.0.0,<4.0.0",
    "pre-commit>=2.20.0,<4.0.0",
    "sphinx>=5.0.0,<8.0.0",
    "sphinx-rtd-theme>=1.2.0,<3.0.0",
]

# ML enhancement dependencies  
ML_REQUIRES = [
    "sentence-transformers>=2.2.0,<3.0.0",
    "torch>=1.12.0,<3.0.0", 
    "scikit-learn>=1.1.0,<2.0.0",
    "pandas>=1.5.0,<3.0.0",
    "matplotlib>=3.5.0,<4.0.0",
    "seaborn>=0.11.0,<1.0.0",
]

# Production deployment dependencies
PROD_REQUIRES = [
    "fastapi>=0.95.0,<1.0.0",
    "uvicorn>=0.18.0,<1.0.0",
    "gunicorn>=20.1.0,<22.0.0",
    "gevent>=22.0.0,<24.0.0",
    "redis>=4.3.0,<6.0.0",
    "sqlalchemy>=1.4.0,<3.0.0",
    "alembic>=1.8.0,<2.0.0",
    "opentelemetry-api>=1.15.0,<2.0.0",
    "opentelemetry-sdk>=1.15.0,<2.0.0",
]

# Cloud deployment dependencies
CLOUD_REQUIRES = [
    "boto3>=1.24.0,<2.0.0",
    "google-cloud-storage>=2.5.0,<3.0.0", 
    "azure-storage-blob>=12.13.0,<13.0.0",
    "celery>=5.2.0,<6.0.0",
    "kombu>=5.2.0,<6.0.0",
]

# Internationalization dependencies
I18N_REQUIRES = [
    "babel>=2.10.0,<3.0.0",
    "gettext>=4.0,<5.0",
]

setup(
    name="quantum-enhanced-rag",
    version="1.0.0",
    description="Quantum-inspired task planning with zero-hallucination RAG capabilities",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    author="Daniel Schmidt",
    author_email="daniel@example.com",  # Update with actual email
    url="https://github.com/danieleschmidt/quantum-inspired-task-planner",
    license="MIT",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    package_data={
        "no_hallucination_rag": [
            "quantum/i18n/locales/**/*.po",
            "quantum/i18n/locales/**/*.json",
            "config/*.yaml",
            "templates/*.html",
        ],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "ml": ML_REQUIRES, 
        "prod": PROD_REQUIRES,
        "cloud": CLOUD_REQUIRES,
        "i18n": I18N_REQUIRES,
        "all": DEV_REQUIRES + ML_REQUIRES + PROD_REQUIRES + CLOUD_REQUIRES + I18N_REQUIRES,
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Command line interfaces
    entry_points={
        "console_scripts": [
            "quantum-rag=no_hallucination_rag.cli:main",
            "quantum-planner=no_hallucination_rag.quantum.quantum_planner:cli_main",
            "quantum-shell=no_hallucination_rag.shell.interactive_shell:main",
            "quantum-compliance=no_hallucination_rag.quantum.quantum_compliance:cli_main",
        ],
    },
    
    # Classification metadata
    classifiers=[
        # Development Status
        "Development Status :: 5 - Production/Stable",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        
        # Topic Classification
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis", 
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Framework
        "Framework :: AsyncIO",
        "Framework :: FastAPI",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        
        # Natural Language
        "Natural Language :: English",
        "Natural Language :: Spanish", 
        "Natural Language :: French",
        "Natural Language :: German",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: Japanese",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "quantum-computing", "task-planning", "rag", "retrieval-augmented-generation",
        "artificial-intelligence", "machine-learning", "quantum-algorithms",
        "superposition", "entanglement", "interference", "quantum-optimization",
        "compliance", "gdpr", "security", "enterprise", "production-ready",
        "zero-hallucination", "factual-ai", "knowledge-management"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://quantum-rag.readthedocs.io/",
        "Source": "https://github.com/danieleschmidt/quantum-inspired-task-planner",
        "Tracker": "https://github.com/danieleschmidt/quantum-inspired-task-planner/issues",
        "Changelog": "https://github.com/danieleschmidt/quantum-inspired-task-planner/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/danieleschmidt",
    },
    
    # ZIP safety
    zip_safe=False,
    
    # Test configuration
    test_suite="tests",
    tests_require=DEV_REQUIRES,
    
    # Additional metadata
    platforms=["any"],
    
    # Package data files
    data_files=[
        ("share/quantum-rag/config", ["config/production.yaml", "config/development.yaml"]),
        ("share/quantum-rag/docs", ["DEPLOYMENT_GUIDE.md", "ARCHITECTURE_OVERVIEW.md"]),
    ],
)