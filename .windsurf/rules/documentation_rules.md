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

#### File Structure
- Maintain `documentation/progress.md` as the single source of truth for project progress
- Follow this structure:
  ```markdown
  # [Project Name] - Implementation Progress

  ## Status Key
  - ✅ **Complete** - Feature is implemented and tested
  - 🔄 **In Progress** - Work has started but is not complete
  - 🔍 **Under Review** - Implementation complete, awaiting review
  - 🐛 **Bug Fixing** - Feature implemented but has known issues
  - 📝 **Planned** - Scheduled but not started
  - ⏸️ **On Hold** - Started but paused for some reason

  ---

  ## Phase X: [Phase Name]

  ### Step X.Y: [Step Name]
  **Status:** [Status Emoji] [Status Text] (YYYY-MM-DD)

  **Implementation Details:**
  - [ ] Task 1
  - [ ] Task 2
  - [ ] Task 3

  **Current Status:**
  - Current progress details
  - Any blockers or issues

  **Next Steps:**
  - Immediate next actions
  - Dependencies

  **Files Created/Modified:**
  - List of affected files
  
  ---
  
  ## Recent Changes
  
  ### [YYYY-MM-DD] [Change Summary]
  - Detail 1
  - Detail 2
  - Detail 3
  ```

#### Update Guidelines
- Update after every significant code change or feature implementation
- Each update must include:
  - Reference to the implementation plan section
  - Specific changes made
  - Current status with dates
  - Next steps
  - Affected files

#### Status Updates
- Be specific about what was completed
- Include relevant metrics or test results when available
- Note any blockers or dependencies
- Keep updates concise but informative
- Update the status emoji and date when status changes

#### Implementation Details
- Document specific changes made
- Include code snippets when helpful
- Reference related issues or PRs
- Note any deviations from the plan

#### File Maintenance
- Keep the file organized by phases and steps
- Use consistent formatting
- Remove or archive completed items when they become irrelevant
- Ensure all links are working

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
