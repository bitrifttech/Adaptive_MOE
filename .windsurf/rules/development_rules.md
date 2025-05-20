---
trigger: always_on
---

# Development Environment Rules for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## Development Environment Setup

### Environment Isolation
- Use virtual environments for dependency isolation
- Create a clean environment using conda or venv
- Document all environment setup steps
- Maintain separate environments for development and testing
- Specify Python version (3.10) consistently

### Dependency Management
- Pin all dependency versions explicitly
- Separate core dependencies from development dependencies
- Document dependency purposes and relationships
- Avoid unnecessary dependencies
- Use requirements.txt for simple dependency management

### Version Control Practices
- Make atomic, focused commits
- Use descriptive commit messages
- Reference implementation plan phases in commit messages
- Keep changes small and focused
- Follow the project's branching strategy

### Development Workflow
- Start with interface definition
- Implement core functionality
- Add error handling
- Write comprehensive tests
- Document the implementation
- Update progress.md

### Local Development Tools
- Configure linters (e.g., flake8, pylint) according to project style
- Use formatters (e.g., black, isort) for consistent code style
- Set up type checking with mypy
- Use pytest for all testing
- Configure editorconfig for consistent editor settings

### Reproducibility
- Use fixed random seeds for reproducible behavior
- Document hardware requirements and expectations
- Specify expected performance characteristics
- Include sample inputs and outputs for verification
- Document any platform-specific considerations

### Development Debugging
- Add appropriate logging at different verbosity levels
- Include debug options in configurable parameters
- Create visualization utilities for model behavior
- Implement step-by-step execution options for complex algorithms
- Add performance profiling capabilities

### Development Convenience
- Create helper scripts for common tasks
- Implement command-line interfaces for interactive testing
- Add configuration presets for different scenarios
- Create sample data generators for testing
- Provide notebooks for exploratory development
