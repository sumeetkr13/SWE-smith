"""
LLM prompts for test-driven bug generation.

This module contains prompts for code reconciliation - updating implementation
to satisfy mutated tests while potentially breaking other tests.
"""

RECONCILIATION_SYSTEM_PROMPT = """You are a code modification expert. Your task is to make SURGICAL changes to implementation code to satisfy a modified test specification.

CRITICAL REQUIREMENTS:
1. The updated code MUST pass the modified test
2. Make MINIMAL, SURGICAL changes - change ONLY the specific lines/logic needed
3. PRESERVE ALL existing code structure:
   - Keep ALL docstrings exactly as they are
   - Keep ALL comments exactly as they are
   - Keep the same imports (do NOT add duplicate imports)
   - Keep the same function/method signatures
   - Keep the same class structure
4. Try to maintain compatibility with other tests, but it's OK if some break (this is desired for creating regression bugs)
5. Do NOT modify the test itself - only the implementation
6. Do NOT add extensive error handling or validation unless specifically required by the mutated test

You will receive:
- The original test
- The modified test (with broader requirements or relaxed constraints)
- The current implementation
- Related tests for context (DO NOT modify these)

Your goal: Make the SMALLEST possible change to pass the modified test. Think of this as a lazy developer who overfits to one test case - they change just enough to make the new test pass, potentially breaking other tests."""


RECONCILIATION_TASK_PROMPT = """## Task

A test has been modified to have different requirements. Update the implementation to satisfy the new test.

### Mutation Applied
**Type:** {mutation_type}
**Description:** {mutation_description}

### Original Test
```python
{original_test}
```

### Modified Test
```python
{mutated_test}
```

### Current Implementation
```python
{current_impl}
```

{related_tests_section}

## Instructions

1. **Analyze the difference:** Compare the original and modified test to identify EXACTLY what changed
2. **Identify the minimal change:** Find the SINGLE line or expression that needs to change
3. **Make surgical modification:** Change ONLY that specific line/expression
4. **Preserve everything else:** Keep all docstrings, comments, imports, and structure EXACTLY as shown

## Output Format

Provide ONLY the updated implementation code in a ```python code block.

**CRITICAL RULES**:
1. You are modifying an EXISTING function/class - provide the complete code for ONLY the entity shown in "Current Implementation"
2. Do NOT create new classes or duplicate existing ones
3. PRESERVE ALL docstrings and comments exactly as they appear
4. Do NOT add new imports - use only what's already imported
5. Make the SMALLEST possible change to pass the mutated test

**Requirements:**
- Complete code for the entity being modified (same name and signature as Current Implementation)
- ALL docstrings preserved verbatim
- ALL comments preserved verbatim
- Same imports as Current Implementation
- MINIMAL logic changes (usually 1-3 lines)
- No explanations outside the code block
- No test code

**Examples of GOOD minimal changes:**

Example 1 - Changing a regex pattern:
```python
class DateTimeParser:
    """Original docstring preserved exactly."""

    _FOUR_DIGIT_RE: ClassVar[Pattern[str]] = re.compile(r"\d{4,}")  # Changed from \d{4}

    def parse(self, date_str):
        """Original method docstring preserved."""
        # Original comment preserved
        return self._FOUR_DIGIT_RE.match(date_str)
```

Example 2 - Relaxing a validation:
```python
def validate_input(x, y):
    """Original docstring preserved exactly."""
    # Only validate x, removed y validation to pass mutated test
    if x < 0:
        raise ValueError("x must be positive")
    return x + y
```

Example 3 - Broadening a condition:
```python
def process_value(val):
    """Process a value within acceptable range."""
    # Changed <= to < to allow boundary value
    if val < 1000:  # Was: val <= 1000
        return val * 2
    return val
```

**BAD examples (DO NOT DO THIS):**
❌ Removing all docstrings
❌ Removing all comments
❌ Adding duplicate imports
❌ Rewriting entire methods when only one line needs to change
❌ Adding new functionality not required by the mutated test

Now provide the updated implementation for the entity shown in Current Implementation:"""


RELATED_TESTS_SECTION = """### Related Tests (for context - DO NOT modify these)
```python
{related_tests}
```

Note: Your changes may cause some of these tests to fail. That's acceptable."""


def format_reconciliation_prompt(
    original_test: str,
    mutated_test: str,
    current_impl: str,
    mutation_type: str,
    mutation_description: str,
    related_tests: str = "",
) -> str:
    """
    Format the reconciliation task prompt.

    Args:
        original_test: Original test code
        mutated_test: Mutated test code
        current_impl: Current implementation code
        mutation_type: Name of mutation type applied
        mutation_description: Description of what was mutated
        related_tests: Other related test code for context

    Returns:
        Formatted prompt string
    """
    # Add related tests section if available
    related_section = ""
    if related_tests and related_tests.strip():
        related_section = RELATED_TESTS_SECTION.format(related_tests=related_tests)

    return RECONCILIATION_TASK_PROMPT.format(
        original_test=original_test,
        mutated_test=mutated_test,
        current_impl=current_impl,
        mutation_type=mutation_type,
        mutation_description=mutation_description,
        related_tests_section=related_section,
    )


# Alternative: More aggressive prompt for creating obvious bugs
AGGRESSIVE_RECONCILIATION_PROMPT = """## Task: Create a Regression Bug

Your goal is to update the implementation to pass the mutated test WHILE INTENTIONALLY BREAKING other tests.

This is for generating training data for AI agents learning to fix bugs.

### Modified Test (MUST PASS)
```python
{mutated_test}
```

### Current Implementation
```python
{current_impl}
```

### Related Tests (Should FAIL after your changes)
```python
{related_tests}
```

## Strategy

Make changes that:
1. ✅ Satisfy the mutated test
2. ❌ Break at least one related test
3. Look like realistic bugs (overfitting to one requirement)

Common patterns:
- Remove validation that the mutated test no longer checks
- Broaden type constraints if mutated test generalizes types
- Remove bounds checking if mutated test uses larger values
- Add special cases that break general behavior

## Output

Provide ONLY the updated Python code:"""


def format_aggressive_prompt(
    mutated_test: str,
    current_impl: str,
    related_tests: str = "",
) -> str:
    """Format aggressive prompt that explicitly aims to break tests."""
    return AGGRESSIVE_RECONCILIATION_PROMPT.format(
        mutated_test=mutated_test,
        current_impl=current_impl,
        related_tests=related_tests or "# No related tests found",
    )


# Mutation type descriptions for better prompting
MUTATION_DESCRIPTIONS = {
    "broaden_input": "Input values were increased/broadened to test larger ranges",
    "relax_assertion": "Assertion operator was relaxed (e.g., == changed to >=)",
    "add_parameter": "Additional parameter was added to function call",
    "modify_edge_case": "Edge case value was changed to near-edge value (e.g., [] to [None])",
    "generalize_type": "Type constraint was generalized (e.g., list to (list, tuple))",
    "expand_scope": "Assertion scope was expanded with additional conditions",
}


def get_mutation_description(mutation_type: str) -> str:
    """Get human-readable description of mutation type."""
    return MUTATION_DESCRIPTIONS.get(
        mutation_type,
        f"Test was modified using {mutation_type} mutation",
    )
