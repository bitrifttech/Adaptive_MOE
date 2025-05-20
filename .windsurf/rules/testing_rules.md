---
trigger: always_on
---

# Testing Rules for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## Testing Conventions

### Test Structure
- Write three types of tests:
  - **Unit Tests**: Test individual components in isolation
  - **Integration Tests**: Test interactions between components
  - **End-to-End Tests**: Test the entire pipeline
- Place tests in the appropriate directory:
  - `tests/unit/` for unit tests
  - `tests/integration/` for integration tests

### Test Naming Conventions
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<method_name>_<scenario>_<expected_behavior>`
- Examples:
  - `test_confidence_calculation_low_input_returns_zero.py`
  - `test_router_high_confidence_skips_experts.py`

### Testing Tools
- Use pytest as the test runner
- Leverage pytest fixtures for common setup/teardown
- Use pytest-cov for measuring test coverage
- Aim for high test coverage (>80%)

### Testing Best Practices
- Follow the Arrange-Act-Assert pattern:
  1. **Arrange**: Set up the test data and environment
  2. **Act**: Call the function/method being tested
  3. **Assert**: Verify the expected outcome
- Test both happy paths and edge cases
- Use mocks or stubs for external dependencies
- Keep tests independent of each other
- Write deterministic tests (same input = same output)
- One assertion per test is preferred

### Feature-First Development
- Write tests before implementing features (TDD approach)
- Ensure every feature has corresponding tests
- Add tests for bug fixes to prevent regressions
- Test all public APIs

### Testing Environment
- Tests should run in isolation
- Avoid tests that depend on external services
- Use environment variables for configuration
- Create test data factories for consistent test data

### Test Documentation
- Include a brief description of what each test is verifying
- Document any complex test setup
- Explain the purpose of test fixtures
- Comment on any non-obvious assertions

### Continuous Testing
- Run tests before committing code
- Ensure all tests pass before requesting a code review
- Update tests when modifying existing functionality
- Refactor tests as needed to improve clarity and maintenance
