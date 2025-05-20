---
trigger: always_on
---

# AI-Assisted Development Guidelines for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## AI-Assisted Development Approach

### Understanding Requirements
- Carefully analyze all requirements before beginning implementation
- Ask clarifying questions when requirements are ambiguous
- Confirm understanding by summarizing requirements before coding
- Reference specific sections of the project documentation when discussing implementation

### Implementation Planning
- Break down complex ML tasks into smaller, manageable steps
- Clearly outline the implementation plan before writing code
- Identify potential challenges and edge cases upfront
- Consider memory usage and computational efficiency in the planning stage

### Modularity and Abstraction
- Implement each component with clear abstraction boundaries
- Document interfaces thoroughly before implementation
- Use type hints consistently for tensor operations
- Create clean separation between model logic and infrastructure code

### Explaining Technical Choices
- Provide clear rationale for algorithm selections
- Explain trade-offs between different implementation approaches
- Justify hyperparameter choices with references or reasoning
- Document any assumptions made about data or model behavior

### Code Generation Standards
- Generate complete, functional code (not pseudo-code)
- Include all necessary imports
- Provide comprehensive docstrings for all functions and classes
- Include inline comments for complex tensor operations or algorithms

### ML-Specific Best Practices
- Initialize weights appropriately for neural network components
- Implement gradient flow control (detach, no_grad) where appropriate
- Follow PyTorch/ML conventions for tensor dimensions and operations
- Be explicit about tensor shapes and types in docstrings and comments

### Testing Machine Learning Components
- Generate test cases that verify numerical correctness
- Include tests for edge cases (empty tensors, extreme values)
- Create reproducible tests with fixed random seeds
- Test both forward and backward passes for neural components

### Debugging Assistance
- Suggest possible debugging strategies for complex components
- Recommend logging points for important tensors and values
- Provide guidance on visualizing model behavior
- Suggest common failure modes to check

### Incremental Verification
- Decompose complex operations into smaller, testable steps
- Verify each step independently before combining
- Suggest intermediate outputs to check for correctness
- Provide expected behavior for each component

### Knowledge Gaps and Alternatives
- Clearly state when approaching less familiar implementation areas
- Provide multiple alternative approaches when appropriate
- Reference relevant research papers or documentation
- Explain the reasoning process for technical decisions

### Code Review Perspective
- Self-review generated code for potential issues
- Highlight areas that might need optimization
- Identify potential numeric stability issues
- Flag code that might be difficult to test
