---
trigger: always_on
---

# Implementation Sequence Rules for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## Implementation Sequence

### Phased Development Approach
- Follow the project's 5-phase implementation plan strictly, following all substeps as well:
  1. Foundation Setup
  2. Router Implementation
  3. Knowledge Gap Detection
  4. Expert Creation Pipeline
  5. Integration Layer
- Do not begin a new phase until the previous phase is fully implemented and tested
- Within each phase, implement components in the specified order
- Update progress.md after completing each step in a phase

### Incremental Implementation
- Implement one logical unit at a time (class, method, or small feature)
- Verify functionality with tests before moving to the next component
- Create atomic, focused commits for each logical unit
- When implementing a complex component:
  1. First create the interfaces/contracts
  2. Implement a minimal working version
  3. Add tests
  4. Refine and optimize only after tests pass

### Component Dependencies
- When a component depends on another, implement dependencies first
- Use stubs or mock objects to test components that depend on unimplemented features
- Document assumptions about how dependent components will interact
- Revisit and update integration points when all components are complete

### Implementation Checkpoints
- After implementing each component, verify:
  - All tests pass
  - Code adheres to style guidelines
  - Documentation is updated
  - The component meets all requirements
- Before starting a new component, ensure the current one is fully functional
- Once the current component is implemented, tests are passing, and documentation is updated. Commit the changes.

### Error Handling Strategy
- Implement basic happy path functionality first
- Add error handling after the basic functionality works
- Test both normal and error paths
- Document expected error scenarios

### Feature Completeness
- For each feature, ensure it meets all requirements before moving on
- Include both functional and non-functional requirements:
  - Correctness (does it work?)
  - Performance (is it efficient?)
  - Robustness (does it handle errors?)
  - Testability (can it be properly tested?)

### Testing Sequence
- Write unit tests for each component as it's implemented
- Add integration tests when multiple components are ready
- Create end-to-end tests for each phase before moving to the next
- Verify regression tests pass after each implementation step

### Refactoring Guidelines
- Prioritize correctness over premature optimization
- Refactor only after tests are in place
- Consider refactoring when:
  - Code duplication is identified
  - A cleaner implementation pattern is recognized
  - Performance issues are detected
- Always maintain test coverage during refactoring