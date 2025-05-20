# Modularity Rules for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## Code Organization and Modularity

### Single Responsibility Principle
- Each module, class, or function should have only one reason to change
- Keep modules focused and small (fewer than 500 lines)
- Separate concerns between different components
- Avoid mixing unrelated functionality in the same file

### Module Structure
- Follow the project's predefined structure exactly:
  - `core/` for base model and pipeline
  - `router/` for routing logic
  - `experts/` for expert implementations
  - `integration/` for response integration
  - `utils/` for utility functions
  - `config/` for configuration files
  - `tests/` for test code
- Each module should have a clear and specific purpose

### Dependency Injection
- Dependencies should be injected rather than created within classes
- Use interfaces/abstract base classes to define contracts
- Make dependencies explicit in constructor parameters
- Avoid creating instances of other classes directly within a class
- This approach makes testing and mocking easier

### Interface Design
- Define clear interfaces using abstract base classes
- Separate interface definition from implementation
- Use descriptive method names that indicate what the method does
- Provide comprehensive docstrings for all interface methods

### Implementation Isolation
- Implement each component independently of others
- Use mock objects for testing
- Ensure components can be tested in isolation
- Minimize coupling between modules

### Configuration Management
- All configurable parameters should be externalized
- Use YAML for configuration files
- Support environment variable overrides
- Keep defaults reasonable and document them clearly
