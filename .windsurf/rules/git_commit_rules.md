---
trigger: always_on
---

# Git Commit Guidelines for Adaptive MoE

## Commit Workflow

### Before Committing
1. **Run Tests**
   - Execute all unit tests: `pytest tests/`
   - Ensure all tests pass before committing
   - If any tests are skipped, document the reason in the test file

2. **Code Quality Checks**
   - Run linters: `pre-commit run --all-files`
   - Fix all linting and formatting issues
   - Ensure type hints are present and accurate
   - Verify docstrings follow Google style guide

3. **Code Review**
   - Self-review your changes with `git diff --staged`
   - Ensure changes are focused and atomic
   - Remove any debug code or temporary comments

### Writing Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Commit Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

#### Commit Message Examples
```
feat(router): implement confidence-based routing

Add confidence threshold configuration and routing logic based on model
confidence scores. Includes comprehensive test coverage for the router.

Resolves: #123
```

```
fix(model): handle edge case in tensor processing

Fix division by zero in attention computation when sequence length is 1.
Add test cases for single-token sequences.

Fixes: #124
```

### Commit Best Practices

1. **Keep Commits Atomic**
   - Each commit should represent a single logical change
   - Avoid mixing unrelated changes in a single commit

2. **Write Descriptive Messages**
   - Use the imperative mood ("Add feature" not "Added feature")
   - First line should be 50 characters or less
   - Include a blank line between the subject and body
   - Wrap the body at 72 characters

3. **Reference Issues**
   - Link to relevant issues using keywords (e.g., Fixes #123)
   - If the commit closes an issue, use the full URL

4. **Document Breaking Changes**
   - Use `BREAKING CHANGE:` in the footer for any breaking changes
   - Include migration instructions if needed

### After Committing
1. **Verify the Commit**
   - Review the commit with `git show`
   - Ensure all intended changes are included

2. **Push Changes**
   - Push to the appropriate branch
   - Create a pull request if working in a team
   - Link to any related issues or pull requests

## Pre-commit Hooks

This project uses pre-commit hooks to enforce code quality. The following hooks are configured:

- `black`: Code formatting
- `isort`: Import sorting
- `flake8`: Linting
- `mypy`: Static type checking

These hooks run automatically on each commit. To run them manually:

```bash
pre-commit run --all-files
```

## Branch Naming Convention

Use the following format for branch names:

```
<type>/<description>-<issue-number>
```

Example:
```
feat/implement-router-123
fix/attention-bug-124
```
