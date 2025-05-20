# Code Style Rules for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## Code Style Conventions

### PEP 8 Compliance
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code
- Use 4 spaces for indentation, not tabs
- Keep line length to 88 characters (Black default)
- Use proper naming conventions:
  - `snake_case` for functions, methods, and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Type Hints
- Use type hints for all function signatures
- Include return type annotations for all functions
- Use the `typing` module for complex types
- For example:
  ```python
  from typing import List, Optional, Dict
  
  def process_data(input_data: List[str], config: Optional[Dict[str, str]] = None) -> bool:
      # Implementation
      return True
  ```

### Docstrings
- Document all public APIs with docstrings
- Follow Google style for docstrings
- Include descriptions for:
  - Function/method purpose
  - Parameters
  - Return values
  - Exceptions raised
- Example:
  ```python
  def calculate_confidence(hidden_states: torch.Tensor) -> float:
      """Calculate confidence score from model hidden states.
      
      Args:
          hidden_states: Tensor containing model hidden states
          
      Returns:
          Confidence score between 0 and 1
          
      Raises:
          ValueError: If hidden_states has incorrect shape
      """
      # Implementation
  ```

### Imports
- Use absolute imports, not relative imports
- Organize imports in three groups, separated by a blank line:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
- Sort imports alphabetically within each group
- Avoid wildcard imports (`from module import *`)

### Code Structure
- One class per file is preferred
- Group related functions together
- Place constants at the top of the file
- Order class methods logically:
  1. Special methods (`__init__`, etc.)
  2. Public methods
  3. Protected methods (prefixed with `_`)
  4. Private methods (prefixed with `__`)

### Comments
- Write comments for complex logic or non-obvious code
- Focus on why, not what (the code shows what it does)
- Keep comments up-to-date with code changes
- Use TODO comments for planned improvements (format: `# TODO: description`)

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Handle exceptions at the appropriate level
- Use context managers (`with` statement) for resource cleanup
