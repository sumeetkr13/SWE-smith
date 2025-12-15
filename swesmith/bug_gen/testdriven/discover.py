"""
Test discovery and analysis for test-driven bug generation.

This module identifies tests suitable for mutation by analyzing:
- Test file structure
- Code coverage (heuristic)
- Test complexity
- Test dependencies
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from swesmith.bug_gen.adapters import get_entities_from_file
from swesmith.constants import CodeEntity
from swesmith.profiles.base import RepoProfile
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestCandidate:
    """
    Represents a test that can be mutated.

    Attributes:
        file_path: Path to test file
        test_name: Name of test function
        test_function: CodeEntity for the test
        covered_entities: Code entities this test covers (heuristic)
        coverage_score: Estimated coverage score (0-1)
        dependencies: Other test names in same file
        assertion_count: Number of assertions in test
        has_parametrize: Whether test is parametrized
    """

    file_path: str
    test_name: str
    test_function: CodeEntity
    covered_entities: list[CodeEntity] = field(default_factory=list)
    coverage_score: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    assertion_count: int = 0
    has_parametrize: bool = False

    def __repr__(self) -> str:
        return f"TestCandidate({self.test_name}, coverage={self.coverage_score:.2f})"


def discover_tests(
    repo: str,
    rp: RepoProfile,
    coverage_threshold: float = 0.3,
    max_tests_per_file: int = -1,
) -> list[TestCandidate]:
    """
    Discover and rank tests suitable for mutation.

    Strategy:
    1. Find all test files using rp.test_paths
    2. Parse test functions using language adapters
    3. Estimate coverage (heuristic: imports, function calls, assertions)
    4. Filter tests with good coverage
    5. Rank by complexity and coverage

    Args:
        repo: Repository path
        rp: RepoProfile instance
        coverage_threshold: Minimum coverage score (0-1)
        max_tests_per_file: Limit tests per file (-1 for unlimited)

    Returns:
        List of TestCandidate objects sorted by suitability
    """
    logger.info(f"Discovering tests in {repo}...")

    # Get test file paths using glob patterns
    # Assume repo is cloned in current directory
    repo_path = Path(repo) if Path(repo).exists() else Path.cwd() / repo

    test_files = []
    # Common test file patterns
    patterns = [
        str(repo_path / "tests" / "test_*.py"),
        str(repo_path / "tests" / "*_test.py"),
        str(repo_path / "test" / "test_*.py"),
        str(repo_path / "test" / "*_test.py"),
        str(repo_path / "**/test_*.py"),
    ]

    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        test_files.extend([Path(f) for f in found])

    # Remove duplicates
    test_files = list(set(test_files))

    if not test_files:
        logger.warning(f"No test files found in {repo}")
        return []

    logger.info(f"Found {len(test_files)} test files")

    candidates = []
    for test_file in test_files:
        try:
            file_candidates = _discover_tests_in_file(
                test_file=test_file,
                repo=repo,
                rp=rp,
                coverage_threshold=coverage_threshold,
            )
            candidates.extend(file_candidates)

            # Limit per file if specified
            if max_tests_per_file > 0:
                candidates = candidates[:max_tests_per_file]

        except Exception as e:
            logger.debug(f"Error processing {test_file}: {e}")
            continue

    # Filter out test classes that are likely to fail entity identification
    # Exclude parser, formatter, and locale tests
    EXCLUDED_TEST_PATTERNS = [
        "Parser",  # TestDateTimeParserParse, TestDateTimeParserISO, etc.
        "Formatter",  # TestFormatterFormatToken, etc.
        "Locale",  # TestTagalogLocale, TestRussianLocale, etc.
    ]

    filtered_candidates = []
    for candidate in candidates:
        test_name = candidate.test_name
        # Check if test name contains any excluded patterns
        if any(pattern in test_name for pattern in EXCLUDED_TEST_PATTERNS):
            logger.debug(f"Excluding test due to pattern match: {test_name}")
            continue
        filtered_candidates.append(candidate)

    candidates = filtered_candidates
    logger.info(
        f"After filtering: {len(candidates)} test candidates (excluded parser/formatter/locale tests)"
    )

    # Sort by coverage score (descending) and assertion count
    candidates = sorted(
        candidates,
        key=lambda x: (x.coverage_score, x.assertion_count),
        reverse=True,
    )

    logger.info(
        f"Discovered {len(candidates)} test candidates (threshold={coverage_threshold})"
    )

    return candidates


def _discover_tests_in_file(
    test_file: Path,
    repo: str,
    rp: RepoProfile,
    coverage_threshold: float,
) -> list[TestCandidate]:
    """Discover tests in a single file."""
    candidates = []

    # Get all entities from test file
    ext = test_file.suffix
    if ext not in rp.exts:
        return []

    try:
        entities = []
        get_entities_from_file[ext](entities, str(test_file))
    except Exception as e:
        logger.debug(f"Could not parse {test_file}: {e}")
        return []

    # Get all test names in this file for dependency tracking
    all_test_names = [e.name for e in entities if _is_test_function(e.name, ext)]

    for entity in entities:
        if not _is_test_function(entity.name, ext):
            continue

        # Analyze test coverage and complexity
        coverage_info = _analyze_test_coverage(entity, repo, rp)

        if coverage_info["score"] < coverage_threshold:
            continue

        # Get dependencies (other tests in same file)
        dependencies = [name for name in all_test_names if name != entity.name]

        candidates.append(
            TestCandidate(
                file_path=str(test_file),
                test_name=entity.name,
                test_function=entity,
                covered_entities=coverage_info["entities"],
                coverage_score=coverage_info["score"],
                dependencies=dependencies,
                assertion_count=coverage_info["assertion_count"],
                has_parametrize=coverage_info["has_parametrize"],
            )
        )

    return candidates


def _is_test_function(name: str, ext: str) -> bool:
    """
    Check if entity name indicates it's a test function.

    Supports:
    - Python: test_*, Test*
    - Go: Test*, Benchmark*
    - Rust: #[test]
    """
    name_lower = name.lower()

    if ext == ".py":
        return name_lower.startswith("test_") or name.startswith("Test")
    elif ext == ".go":
        return name.startswith("Test") or name.startswith("Benchmark")
    elif ext == ".rs":
        # Rust tests are marked with #[test] attribute
        # This is handled in the adapter
        return "test" in name_lower

    return False


def _analyze_test_coverage(entity: CodeEntity, repo: str, rp: RepoProfile) -> dict:
    """
    Heuristically estimate what code a test covers.

    Heuristics:
    1. Count assertions (more = better coverage)
    2. Count function calls (more = more code exercised)
    3. Check for parametrize (indicates comprehensive testing)
    4. Identify imported modules/functions

    Returns:
        {
            'score': float (0-1),
            'entities': list[CodeEntity],
            'assertion_count': int,
            'has_parametrize': bool,
        }
    """
    src_code = entity.src_code

    # Count assertions
    assertion_count = src_code.count("assert")

    # Count function calls (rough heuristic)
    function_call_count = src_code.count("(") - src_code.count("def ")

    # Check for parametrize decorators
    has_parametrize = (
        "@pytest.mark.parametrize" in src_code or "@parametrize" in src_code
    )

    # Heuristic scoring
    score = 0.0

    # Assertions contribute most
    if assertion_count > 0:
        score += min(assertion_count * 0.2, 0.6)  # Cap at 0.6

    # Function calls indicate complexity
    if function_call_count > 2:
        score += min(function_call_count * 0.05, 0.3)  # Cap at 0.3

    # Parametrized tests are valuable
    if has_parametrize:
        score += 0.2

    # Normalize to 0-1
    score = min(score, 1.0)

    # Try to identify covered entities (simplified for now)
    covered_entities = _identify_covered_entities(entity, repo, rp)

    return {
        "score": score,
        "entities": covered_entities,
        "assertion_count": assertion_count,
        "has_parametrize": has_parametrize,
    }


def _identify_covered_entities(
    test_entity: CodeEntity, repo: str, rp: RepoProfile
) -> list[CodeEntity]:
    """
    Identify code entities covered by this test.

    Strategy:
    1. Parse imports to find implementation modules AND specific imported names
    2. Extract function calls from test body
    3. Use test name heuristics (test_foo -> foo)
    4. Match with priority scoring:
       - Priority 10: Explicitly imported names (from X import Y)
       - Priority 5: Generic function calls or test name inference
    5. Filter out excluded entities (parsers, formatters, etc.)
    6. Return top-ranked matched CodeEntity objects

    Args:
        test_entity: The test function entity
        repo: Repository path
        rp: RepoProfile instance

    Returns:
        List of CodeEntity objects that this test covers (up to 3)
    """
    # Exclude complex classes that are hard to mutate correctly
    EXCLUDED_CLASSES = {
        "DateTimeParser",
        "TzinfoParser",
        "ArrowFormatter",
        "Formatter",
    }

    # Exclude files that contain complex parsing/formatting logic
    EXCLUDED_FILE_PATTERNS = [
        "parser.py",
        "formatter.py",
    ]
    import ast
    import re
    from pathlib import Path

    covered = []

    try:
        # Get the test file extension to determine language
        test_file_path = Path(test_entity.file_path)
        ext = test_file_path.suffix

        # Currently only support Python
        if ext != ".py":
            logger.debug(f"Coverage analysis not yet supported for {ext}")
            return []

        # 1. Parse imports to get both files and specific imported names
        impl_files, imported_names = _extract_imports_with_names(
            test_entity.file_path, repo
        )

        logger.info(
            f"[{test_entity.name}] Found {len(impl_files)} impl files, "
            f"{len(imported_names)} imported names: {imported_names}"
        )

        # 1b. Add fallback files for common patterns if no files found
        if not impl_files:
            repo_path = Path(repo) if Path(repo).exists() else Path.cwd() / repo
            test_file = Path(test_entity.file_path)

            # For arrow repository specifically
            if "arrow" in repo:
                # test_arrow.py -> arrow/arrow.py
                # test_factory.py -> arrow/factory.py
                # test_api.py -> arrow/api.py
                test_filename = test_file.stem  # e.g., "test_arrow"
                if test_filename.startswith("test_"):
                    impl_name = test_filename[5:]  # Remove "test_" prefix
                    impl_path = repo_path / "arrow" / f"{impl_name}.py"
                    if impl_path.exists():
                        impl_files.append(str(impl_path))
                        logger.info(f"[{test_entity.name}] Added fallback file: {impl_path}")

        # 2. Extract function/class calls from test body
        called_names = _extract_called_names(test_entity)

        # 3. Add test name heuristics
        test_name = test_entity.name

        # Heuristic 1: test_foo -> foo
        if test_name.startswith("test_"):
            inferred_name = test_name[5:]  # Remove "test_" prefix
            called_names.add(inferred_name)

        # Heuristic 2: Get containing class name for class-based tests
        # TestTagalogLocale -> TagalogLocale
        test_class_name = _get_test_class_name(test_entity)
        if test_class_name and test_class_name.startswith("Test"):
            inferred_class = test_class_name[4:]  # Remove "Test" prefix
            called_names.add(inferred_class)
            imported_names.add(inferred_class)  # Boost priority

            # Heuristic 3: For method-based test classes, also try base class
            # TestArrowHumanize -> Try both "ArrowHumanize" and "Arrow"
            # TestArrowShift -> Try both "ArrowShift" and "Arrow"
            for common_base in ["Arrow", "Factory", "Api"]:
                if inferred_class.startswith(common_base):
                    called_names.add(common_base)
                    imported_names.add(common_base)  # Boost priority
                    logger.debug(
                        f"Added base class heuristic: {common_base} for {test_class_name}"
                    )

        # Prioritize explicitly imported names over generic calls
        priority_names = imported_names
        secondary_names = called_names - imported_names

        # 4. For each implementation file, get entities and match with priority
        for impl_file in impl_files:
            entities = []
            try:
                get_entities_from_file[ext](entities, impl_file)
                logger.info(
                    f"[{test_entity.name}] Found {len(entities)} entities in {impl_file}: "
                    f"{[e.name for e in entities[:5]]}"
                )
            except Exception as e:
                logger.warning(f"Could not parse {impl_file}: {e}")
                continue

            # Match entities by name with priority scoring
            for entity in entities:
                # Skip exception/error classes - they're usually imported but not the target
                if entity.name.endswith(("Error", "Exception", "Warning")):
                    logger.debug(f"Skipping exception class: {entity.name}")
                    continue

                # Skip excluded classes (parsers, formatters, etc.)
                if entity.name in EXCLUDED_CLASSES:
                    logger.debug(f"Skipping excluded class: {entity.name}")
                    continue

                # Skip entities in excluded files
                if any(pattern in entity.file_path for pattern in EXCLUDED_FILE_PATTERNS):
                    logger.debug(f"Skipping entity in excluded file: {entity.name} in {entity.file_path}")
                    continue

                # Boost priority if name matches test class inference
                priority = 0
                if entity.name in priority_names:
                    priority = 10  # High priority
                    # Extra boost if it matches the test class name
                    if test_class_name and entity.name == test_class_name[4:]:
                        priority = 15  # Highest priority
                elif entity.name in secondary_names:
                    priority = 5  # Lower priority

                if priority > 0:
                    covered.append((entity, priority))

        # Sort by priority (highest first) and deduplicate
        covered.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_covered = []

        for entity, priority in covered:
            key = (entity.file_path, entity.name)
            if key not in seen:
                seen.add(key)
                unique_covered.append(entity)

        # Limit to top 3 matches
        unique_covered = unique_covered[:3]

        if unique_covered:
            logger.debug(
                f"Found {len(unique_covered)} covered entities for {test_entity.name}: "
                f"{[e.name for e in unique_covered]}"
            )

        return unique_covered

    except Exception as e:
        logger.debug(f"Error identifying covered entities for {test_entity.name}: {e}")
        return []


def _extract_imports_with_names(
    test_file_path: str, repo: str
) -> tuple[list[str], set[str]]:
    """
    Extract implementation file paths AND specific imported names from a test file.

    This is crucial for priority scoring - explicitly imported names like
    "from arrow.parser import DateTimeParser" should be prioritized over
    generic function calls in the test body.

    Args:
        test_file_path: Path to the test file
        repo: Repository path

    Returns:
        Tuple of (list of file paths, set of imported names)

    Example:
        Input: "from arrow.parser import DateTimeParser, ParserError"
        Output: (["arrow/parser.py"], {"DateTimeParser", "ParserError"})
    """
    import ast
    from pathlib import Path

    impl_files = []
    imported_names = set()

    try:
        with open(test_file_path, "r") as f:
            test_content = f.read()

        tree = ast.parse(test_content)
        test_dir = Path(test_file_path).parent
        repo_path = Path(repo) if Path(repo).exists() else Path.cwd() / repo

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # from arrow.parser import DateTimeParser, ParserError
                if node.module:
                    module_path = _resolve_module_to_file(
                        node.module, repo_path, test_dir
                    )
                    if module_path:
                        impl_files.append(module_path)

                        # Track specific imported names for priority scoring
                        for alias in node.names:
                            if alias.name != "*":
                                imported_names.add(alias.name)

            elif isinstance(node, ast.Import):
                # import arrow.parser
                for alias in node.names:
                    module_path = _resolve_module_to_file(
                        alias.name, repo_path, test_dir
                    )
                    if module_path:
                        impl_files.append(module_path)
                        # For "import X", we don't add to imported_names
                        # since the test likely calls X.something()

    except Exception as e:
        logger.debug(f"Could not extract imports from {test_file_path}: {e}")

    return list(set(impl_files)), imported_names


def _resolve_module_to_file(
    module_name: str, repo_path: Path, test_dir: Path
) -> str | None:
    """
    Resolve a Python module name to an actual file path.

    Args:
        module_name: e.g., "arrow.parser"
        repo_path: Root path of the repository
        test_dir: Directory containing the test file

    Returns:
        Absolute path to the module file, or None if not found
    """
    # Skip standard library and test modules
    skip_modules = {
        "pytest",
        "unittest",
        "mock",
        "typing",
        "os",
        "sys",
        "datetime",
        "json",
        "re",
        "pathlib",
        "collections",
        "itertools",
        "functools",
        "operator",
        "io",
        "copy",
    }

    # Get the top-level module name
    top_level = module_name.split(".")[0]
    if top_level in skip_modules:
        return None

    # Convert module path to file path
    # e.g., arrow.parser -> arrow/parser.py
    module_parts = module_name.split(".")

    # Try multiple possible locations
    possible_paths = [
        # 1. Relative to repo root: arrow/parser.py
        repo_path / "/".join(module_parts[:-1]) / f"{module_parts[-1]}.py",
        # 2. As package: arrow/parser/__init__.py
        repo_path / "/".join(module_parts) / "__init__.py",
        # 3. Direct file: arrow.py
        repo_path / f"{module_name.replace('.', '/')}.py",
        # 4. Relative to test directory
        test_dir / f"{module_name.replace('.', '/')}.py",
    ]

    for path in possible_paths:
        if path.exists() and path.is_file():
            # Skip test utility files (but allow implementation in test dirs)
            filename = path.name
            path_parts = path.parts

            # Only skip if it's clearly a test utility file
            is_test_utility = (
                filename.startswith("test_")
                or filename.startswith("_test")
                or filename in ["conftest.py", "utils.py", "helpers.py", "fixtures.py"]
            ) and any(part in ["test", "tests", "testing"] for part in path_parts)

            if is_test_utility:
                logger.debug(f"Skipping test utility file: {path}")
                continue

            return str(path)

    return None


def _get_test_class_name(test_entity: CodeEntity) -> str | None:
    """
    Get the name of the class containing this test function.

    For class-based tests like:
        class TestTagalogLocale:
            def test_format(self): ...

    Returns "TestTagalogLocale"

    Args:
        test_entity: The test function entity

    Returns:
        Class name if test is in a class, None otherwise
    """
    import ast

    try:
        # Read the test file
        with open(test_entity.file_path, "r") as f:
            content = f.read()

        tree = ast.parse(content)

        # Find the class containing this function
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this class contains our test function
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == test_entity.name:
                        return node.name

    except Exception as e:
        logger.debug(f"Could not get test class name: {e}")

    return None


def _extract_called_names(test_entity: CodeEntity) -> set[str]:
    """
    Extract names of functions/classes called in the test body.

    Args:
        test_entity: The test function entity

    Returns:
        Set of function/class names called in the test
    """
    import ast

    called_names = set()

    try:
        # Parse the test function code
        tree = ast.parse(test_entity.src_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Extract function name from Call node
                if isinstance(node.func, ast.Name):
                    # Simple call: foo()
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.foo() or module.foo()
                    # We want the method name
                    called_names.add(node.func.attr)
                    # Also try to get the object name for module.function() cases
                    if isinstance(node.func.value, ast.Name):
                        # If it's module.function(), we care about "function"
                        # but also track the module name
                        pass  # We already added the attr above

    except Exception as e:
        logger.debug(f"Could not extract function calls from test: {e}")

    return called_names


def get_related_tests(test_file: str, test_name: str) -> list[str]:
    """
    Get other tests in the same file that might be affected by mutations.

    Args:
        test_file: Path to test file
        test_name: Name of the test function

    Returns:
        List of related test names
    """
    try:
        ext = Path(test_file).suffix
        entities = []
        get_entities_from_file[ext](entities, test_file)

        related = [
            e.name
            for e in entities
            if _is_test_function(e.name, ext) and e.name != test_name
        ]

        return related
    except Exception as e:
        logger.debug(f"Could not get related tests from {test_file}: {e}")
        return []


def filter_by_coverage(
    candidates: list[TestCandidate], threshold: float = 0.5
) -> list[TestCandidate]:
    """Filter test candidates by coverage score."""
    return [c for c in candidates if c.coverage_score >= threshold]


def filter_by_assertions(
    candidates: list[TestCandidate], min_assertions: int = 1
) -> list[TestCandidate]:
    """Filter test candidates by assertion count."""
    return [c for c in candidates if c.assertion_count >= min_assertions]


def rank_by_complexity(candidates: list[TestCandidate]) -> list[TestCandidate]:
    """
    Rank test candidates by complexity.

    Complexity factors:
    - Coverage score
    - Assertion count
    - Parametrization
    """
    return sorted(
        candidates,
        key=lambda x: (
            x.coverage_score,
            x.assertion_count,
            x.has_parametrize,
        ),
        reverse=True,
    )
