---
trigger: always_on
---

# Documentation Update Rules for Adaptive MoE Project

## Core Guidelines
1. Only do exactly as asked, nothing more nothing less.
2. Respect all code guidelines and project requirements in the documentation.
3. Implement a feature and then test the implementation always.

## Documentation Update Conventions

### Progress Tracking

- Update `documentation/progress.md` after every code change or feature implementation
- Each update should include:
  - Date and time of the update
  - Reference to the relevant section in the implementation plan
  - Summary of changes made
  - Current implementation status
  - Next steps or pending items

- Format for progress updates:
  ```markdown
  ## [Date] - [Feature/Component Name]
  
  **Implementation Plan Reference:** Phase X, Step Y
  
  **Changes Made:**
  - Implemented [specific functionality]
  - Added [component/class/method]
  - Modified [existing component] to support [new functionality]
  
  **Current Status:**
  - Phase X is [percentage]% complete
  - [Component] is ready for testing
  - [Feature] is fully implemented and tested
  
  **Next Steps:**
  - Implement [next feature]
  - Test [pending functionality]
  - Address [known limitations]
  ```

- Each entry should clearly indicate which phase and step of the implementation plan is being addressed
- Include specific details about what was implemented, not just generic descriptions
- Be honest about implementation status and note any deviations from the plan
- When multiple components are affected, organize the update by component

### Bug Tracking

- Record all bugs and their fixes in `documentation/bugs.md`
- Each bug entry should include:
  - Date discovered
  - Brief but descriptive bug title
  - Detailed description of the issue
  - Steps to reproduce
  - Root cause analysis
  - Solution implemented
  - Preventive measures (if applicable)

- Format for bug entries:
  ```markdown
  ## [Date] - [Bug Title]
  
  **Description:**
  Brief explanation of the bug and its impact.
  
  **Steps to Reproduce:**
  1. [First step]
  2. [Second step]
  3. [Third step]
  
  **Related Components:**
  - [Component 1]
  - [Component 2]
  
  **Root Cause:**
  Explanation of what caused the bug.
  
  **Solution:**
  Description of how the bug was fixed.
  
  **Prevention:**
  Steps taken to prevent similar bugs in the future.
  ```

- Categorize bugs by component or module when possible
- Include code snippets when relevant
- Record bugs even if they seem minor
- If the same bug appears in multiple places, note all occurrences
- Include any tests added to prevent regression

### General Documentation Practices

- Keep documentation updated in sync with code changes
- Use clear, concise language
- Include code examples where helpful
- Maintain a consistent formatting style
- Link related documentation sections when appropriate
- Review documentation regularly for accuracy

### Implementation Status Indicators

Use the following status indicators for clarity:

- ✅ **Complete** - Feature is implemented and tested
- 🔄 **In Progress** - Work has started but is not complete
- 🔍 **Under Review** - Implementation complete, awaiting review
- 🐛 **Bug Fixing** - Feature implemented but has known issues
- 📝 **Planned** - Scheduled but not started
- ⏸️ **On Hold** - Started but paused for some reason
