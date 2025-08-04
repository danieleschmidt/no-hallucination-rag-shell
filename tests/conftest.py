"""
Pytest configuration and shared fixtures.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test data."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Mock response"))]
    )
    return mock_client


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "title": "AI Safety Guidelines",
            "content": "AI systems must undergo comprehensive safety testing before deployment. This includes red-team exercises, adversarial testing, and safety benchmarks.",
            "url": "https://example.com/ai-safety",
            "date": "2023-10-30",
            "type": "guidelines",
            "authority_score": 0.95
        },
        {
            "id": "doc_2",
            "title": "NIST AI Risk Management Framework",
            "content": "The NIST framework provides guidance for managing AI risks. Organizations must assess, categorize, and mitigate potential harms from AI systems.",
            "url": "https://nist.gov/ai-framework",
            "date": "2025-01-15",
            "type": "framework",
            "authority_score": 0.98
        },
        {
            "id": "doc_3",
            "title": "EU AI Act Compliance",
            "content": "The European Union AI Act establishes comprehensive rules for AI systems. High-risk AI applications require conformity assessments and CE marking.",
            "url": "https://ec.europa.eu/ai-act",
            "date": "2024-06-01",
            "type": "regulation",
            "authority_score": 0.92
        }
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What are the AI safety testing requirements?",
        "How do I comply with NIST AI framework?",
        "What are the penalties for AI Act violations?",
        "Who needs to undergo AI bias audits?",
        "What is required for AI transparency reporting?"
    ]


@pytest.fixture
def mock_environment_variables():
    """Set up mock environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def disable_logging():
    """Disable logging during tests to reduce noise."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# Async test markers
pytest.mark.asyncio = pytest.mark.asyncio


# Test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security


# Skip markers for conditional tests
pytest.mark.slow = pytest.mark.slow
pytest.mark.requires_network = pytest.mark.requires_network
pytest.mark.requires_gpu = pytest.mark.requires_gpu