# Adaptive MoE Project - Bug Tracking

This document records bugs encountered during the implementation of the Adaptive MoE project and their resolutions. The purpose is to document issues for future reference and to prevent recurring problems.

## Bug Template

```markdown
## [YYYY-MM-DD] - [Bug Title]

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

## Known Issues

<!-- No bugs have been recorded yet. The first entries will appear as issues are encountered during implementation. -->

<!-- Example:
## 2025-05-20 - Router Confidence Score Always Returns Zero

**Description:**
The uncertainty router's confidence estimation was always returning 0 regardless of input, causing all queries to be routed to experts.

**Steps to Reproduce:**
1. Initialize the UncertaintyRouter with default settings
2. Pass any hidden states tensor to the forward method
3. Observe that confidence score is always 0.0

**Related Components:**
- router/uncertainty_router.py
- UncertaintyRouter.forward method

**Root Cause:**
The sigmoid activation function was being applied twice - once in the network definition and again in the forward method.

**Solution:**
Removed the redundant sigmoid application in the forward method, keeping only the one in the network definition.

**Prevention:**
- Added unit tests specifically checking for varied confidence scores
- Added comments explaining the network's activation functions
- Updated documentation to clarify the expected output range
-->
