# Adaptive MoE - Coding Guidelines

## Project Structure

```
adaptive_moe/
├── core/                  # Core system components
│   ├── __init__.py
│   ├── base_model.py      # Base model wrapper and utilities
│   └── pipeline.py        # Main execution pipeline
├── router/                # Router implementation
│   ├── __init__.py
│   ├── base_router.py     # Base router interface
│   └── uncertainty_router.py  # Confidence-based router implementation
├── experts/               # Expert-related code
│   ├── __init__.py
│   ├── base_expert.py     # Base expert interface
│   ├── expert_factory.py  # Factory for creating experts
│   └── lora_expert.py     # LoRA-based expert implementation
├── integration/           # Integration layer
│   ├── __init__.py
│   └── response_integrator.py  # Combines outputs from multiple experts
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logging_utils.py   # Logging configuration
│   └── metrics.py         # Performance metrics and monitoring
├── config/                # Configuration files
│   └── default.yaml
├── tests/                 # Test suite
│   ├── unit/              # Unit tests
│   │   ├── test_router.py
│   │   ├── test_experts.py
│   │   └── ...
│   └── integration/       # Integration tests
│       ├── test_pipeline.py
│       └── ...
├── main.py                # Entry point
└── requirements-dev.txt   # Development dependencies
```

## Code Organization Principles

1. **Single Responsibility Principle (SRP)**
   - Each module/class should have only one reason to change
   - Keep modules focused and small (ideally < 500 lines)
   - Separate concerns between different components

2. **Dependency Injection**
   - Dependencies should be injected rather than created within classes
   - Use interfaces/abstract base classes to define contracts
   - Makes testing and mocking easier

3. **Configuration Management**
   - All configurable parameters should be externalized
   - Use YAML for configuration files
   - Support environment variable overrides

## Testing Strategy

### 1. Test Structure
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test the entire pipeline

### 2. Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<method_name>_<scenario>_<expected_behavior>`

### 3. Testing Best Practices
- Use pytest as the test runner
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common test setup/teardown
- Aim for high test coverage (>80%)
- Test both happy paths and edge cases
- Use property-based testing where applicable

## Development Workflow

1. **Feature Development**
   - Create a new branch for each feature
   - Write tests first (TDD approach)
   - Implement the minimal code to pass tests
   - Refactor while keeping tests green

2. **Code Review**
   - All changes must be reviewed before merging
   - At least one approval required
   - All tests must pass

3. **Continuous Integration**
   - Run tests on every push
   - Enforce code style (black, isort, flake8)
   - Generate coverage reports

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all function signatures
- Document public APIs with docstrings (Google style)
- Keep line length to 88 characters (Black default)
- Use absolute imports

## Logging

- Use the built-in `logging` module
- Different log levels for different environments (DEBUG in development, INFO/WARNING in production)
- Structured logging for better analysis
- Include request/context IDs for tracing

## Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors with sufficient context
- Implement proper cleanup in finally blocks

## Documentation

- Module-level docstrings explaining purpose and usage
- Class and method docstrings following Google style
- Update README.md when adding new features
- Document any non-obvious design decisions

## Dependencies

- Pin all dependencies with exact versions
- Separate development and production dependencies
- Use `requirements-dev.txt` for development tools
- Document any system-level dependencies

## Version Control

- Write clear, concise commit messages
- Use feature branches
- Rebase before merging to main
- Squash related commits before merging

## Performance Considerations

- Profile before optimizing
- Use appropriate data structures
- Be mindful of memory usage with large models
- Implement caching where appropriate

## Security

- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate all inputs
- Follow the principle of least privilege

## Example Module Structure

```python
"""
Module: adaptive_moe/router/uncertainty_router.py

Implements a confidence-based router for determining when to use experts.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from adaptive_moe.router.base_router import BaseRouter


@dataclass
class RouterConfig:
    """Configuration for the UncertaintyRouter.
    
    Attributes:
        hidden_size: Size of the hidden layer in the router
        confidence_threshold: Threshold below which to consider a response uncertain
        dropout: Dropout probability for regularization
    """
    hidden_size: int = 768
    confidence_threshold: float = 0.7
    dropout: float = 0.1


class UncertaintyRouter(BaseRouter):
    """A router that uses confidence estimation to determine when to use experts.
    
    This router estimates the confidence of the base model's response and decides
    whether to create/use an expert based on a confidence threshold.
    """
    
    def __init__(self, config: RouterConfig):
        """Initialize the UncertaintyRouter.
        
        Args:
            config: Configuration for the router
        """
        super().__init__()
        self.config = config
        self.confidence_estimator = self._build_confidence_network()
        
    def _build_confidence_network(self) -> nn.Module:
        """Build the neural network for confidence estimation."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Forward pass through the router.
        
        Args:
            hidden_states: Hidden states from the base model
            
        Returns:
            A tuple of (confidence_score, should_use_expert)
        """
        last_hidden = hidden_states[:, -1, :]
        confidence = self.confidence_estimator(last_hidden).squeeze(-1)
        use_expert = confidence < self.config.confidence_threshold
        return confidence, use_expert
```

## Testing Example

```python
"""Tests for the uncertainty router."""

import pytest
import torch

from adaptive_moe.router.uncertainty_router import UncertaintyRouter, RouterConfig


class TestUncertaintyRouter:
    @pytest.fixture
    def router_config(self):
        return RouterConfig(
            hidden_size=64,
            confidence_threshold=0.7,
            dropout=0.0
        )
    
    @pytest.fixture
    def router(self, router_config):
        return UncertaintyRouter(router_config)
    
    def test_forward_returns_confidence_and_decision(self, router):
        """Test that forward pass returns confidence and expert decision."""
        # Arrange
        batch_size = 2
        seq_len = 10
        hidden_size = 64
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Act
        confidence, use_expert = router(hidden_states)
        
        # Assert
        assert confidence.shape == (batch_size,)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0
        assert isinstance(use_expert, torch.Tensor)
        assert use_expert.dtype == torch.bool
    
    def test_uses_expert_when_confidence_below_threshold(self, router):
        """Test that expert is used when confidence is below threshold."""
        # Arrange
        batch_size = 1
        seq_len = 1
        hidden_size = 64
        hidden_states = torch.zeros(batch_size, seq_len, hidden_size)  # Should give ~0.5 confidence
        
        # Act
        confidence, use_expert = router(hidden_states)
        
        # Assert
        assert confidence.item() < 0.7  # Below threshold
        assert use_expert.item() is True
```

## Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Run Tests**
   ```bash
   pytest tests/ -v --cov=adaptive_moe --cov-report=term-missing
   ```

3. **Run Linters**
   ```bash
   black .
   isort .
   flake8
   mypy .
   ```

4. **Run the Application**
   ```bash
   python -m adaptive_moe.main
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]
