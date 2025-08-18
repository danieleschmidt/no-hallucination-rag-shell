# Contributing to Quantum-Enhanced No-Hallucination RAG

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Docker and Docker Compose
- Basic understanding of RAG systems and quantum computing concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/no-hallucination-rag.git
   cd no-hallucination-rag
   ```

2. **Create Development Environment**
   ```bash
   # Using conda
   conda create -n no-hallucination python=3.9
   conda activate no-hallucination
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest
   ```

5. **Start Development Server**
   ```bash
   docker-compose up -d
   python -m no_hallucination_rag.shell.cli
   ```

## Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

### Creating a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Commit Message Format
We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(quantum): add superposition task planning
fix(retrieval): handle empty query edge case
docs(api): update authentication examples
```

### Pull Request Process

1. **Create Pull Request**
   - Use the provided PR template
   - Link to related issues
   - Provide clear description of changes

2. **Code Review Requirements**
   - At least one approving review
   - All CI checks must pass
   - Test coverage must not decrease

3. **Merge Strategy**
   - Squash and merge for feature branches
   - Regular merge for release branches
   - Rebase and merge for hotfixes

## Code Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters

### Type Hints
- All public functions must have type hints
- Use `typing` module for complex types
- Example:
  ```python
  from typing import List, Optional, Dict, Any
  
  def process_query(
      query: str, 
      max_results: int = 10,
      filters: Optional[Dict[str, Any]] = None
  ) -> List[str]:
      pass
  ```

### Documentation
- All modules must have docstrings
- All public functions must have docstrings
- Use Google-style docstrings:
  ```python
  def verify_factuality(claim: str, sources: List[str]) -> float:
      """Verify the factuality of a claim against sources.
      
      Args:
          claim: The claim to verify
          sources: List of source documents
          
      Returns:
          Factuality score between 0.0 and 1.0
          
      Raises:
          ValueError: If claim is empty
      """
      pass
  ```

### Testing Standards

#### Test Structure
```
tests/
├── unit/                 # Unit tests
│   ├── core/
│   ├── quantum/
│   └── retrieval/
├── integration/          # Integration tests
│   ├── api/
│   └── pipeline/
├── e2e/                 # End-to-end tests
└── conftest.py          # Pytest configuration
```

#### Test Requirements
- Minimum 80% code coverage
- Test file naming: `test_<module_name>.py`
- Test function naming: `test_<function_name>_<scenario>`
- Use descriptive assertions with custom messages

#### Example Test
```python
import pytest
from no_hallucination_rag.core.factual_rag import FactualRAG

class TestFactualRAG:
    def test_query_with_valid_input_returns_response(self):
        """Test that valid queries return proper responses."""
        rag = FactualRAG()
        response = rag.query("What is quantum computing?")
        
        assert response.answer is not None, "Response should contain an answer"
        assert response.factuality_score > 0.0, "Should have factuality score"
        assert len(response.sources) > 0, "Should have supporting sources"
    
    def test_query_with_empty_input_raises_error(self):
        """Test that empty queries raise appropriate errors."""
        rag = FactualRAG()
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag.query("")
```

### Security Guidelines

#### Code Security
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user inputs
- Use parameterized queries for database access

#### Dependency Management
- Pin dependency versions in requirements files
- Regular security scans with `safety`
- Update dependencies following security advisories

#### Data Privacy
- Implement PII detection and masking
- Follow GDPR compliance requirements
- Audit log all data access

## Contribution Areas

### High-Priority Areas
1. **Factuality Models**: Improving accuracy of factuality detection
2. **Quantum Algorithms**: Optimizing quantum-inspired components
3. **Performance**: Reducing latency and improving throughput
4. **Documentation**: User guides and API documentation
5. **Testing**: Expanding test coverage and benchmarks

### Getting Started Contributions
- Fix typos in documentation
- Add examples to existing functions
- Improve error messages
- Add unit tests for untested functions

### Advanced Contributions
- Implement new factuality detection models
- Add support for new data sources
- Optimize quantum planning algorithms
- Integrate with new vector databases

## Issue Management

### Bug Reports
Use the bug report template and include:
- Environment details (OS, Python version, dependencies)
- Steps to reproduce
- Expected vs actual behavior
- Minimal code example
- Error messages and stack traces

### Feature Requests
Use the feature request template and include:
- Problem description
- Proposed solution
- Alternative solutions considered
- Use cases and benefits

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Documentation improvements
- `good-first-issue`: Good for newcomers
- `help-wanted`: Extra attention needed
- `priority-high`: Urgent issues

## Development Tools

### Recommended IDEs
- **VSCode**: With Python, Pylance, and GitLens extensions
- **PyCharm**: Professional or Community edition
- **Vim/Neovim**: With appropriate Python plugins

### Useful Commands
```bash
# Code formatting
black .
isort .

# Type checking
mypy no_hallucination_rag/

# Linting
flake8 no_hallucination_rag/

# Security scanning
bandit -r no_hallucination_rag/
safety check

# Test coverage
pytest --cov=no_hallucination_rag --cov-report=html

# Documentation building
cd docs && make html
```

### Pre-commit Hooks
We use pre-commit hooks to ensure code quality:
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Performance Guidelines

### Optimization Priorities
1. **Correctness**: Always prioritize correctness over performance
2. **Readability**: Maintain readable code even when optimizing
3. **Profiling**: Profile before optimizing
4. **Measurement**: Measure improvements with benchmarks

### Common Optimizations
- Use appropriate data structures
- Cache expensive computations
- Minimize I/O operations
- Use vectorized operations with NumPy
- Leverage async/await for concurrent operations

## Documentation Guidelines

### User Documentation
- Write for your audience (beginners vs experts)
- Provide complete, runnable examples
- Include common use cases and edge cases
- Keep documentation up-to-date with code changes

### API Documentation
- Follow OpenAPI 3.0 specification
- Include request/response examples
- Document error codes and responses
- Provide SDK examples in multiple languages

### Architecture Documentation
- Use diagrams to illustrate complex concepts
- Document design decisions and trade-offs
- Keep architecture docs current with implementation
- Include performance characteristics

## Communication

### Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: security@project.com for security issues

### Response Times
- Bug reports: 48 hours
- Feature requests: 1 week
- Security issues: 24 hours
- Pull requests: 1 week

## Recognition

### Contributors
All contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

### Contribution Types
We recognize all types of contributions:
- Code contributions
- Documentation improvements
- Bug reports and testing
- Community support
- Design and UX improvements

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search GitHub issues and discussions
3. Create a new discussion thread
4. Contact maintainers directly for urgent matters

Thank you for contributing to the future of trustworthy AI!