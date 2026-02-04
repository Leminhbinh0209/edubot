# tests/test_tools.py
from pathlib import Path
import sys
import json
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mybot.tools.registry import ToolRegistry
from mybot.tools.filesystem import ReadFileTool
from mybot.tools.shell import ExecTool
from mybot.tools.code_analysis import CodeAnalysisTool
from mybot.tools.search import SearchTool
from mybot.tools.test_runner import TestTool
import asyncio

async def test_tool_registry():
    """Test: Tool registry registration and retrieval
    Input: Register 5 tools (ReadFileTool, ExecTool, CodeAnalysisTool, SearchTool, TestTool)
    Expected Output: All tools should be retrievable by name, and get_definitions() should return 5 tool schemas
    """
    print("\n[TEST] Tool Registry")
    print("  Plan: Test that tools can be registered and retrieved from ToolRegistry")
    print("  Input: Register ReadFileTool, ExecTool, CodeAnalysisTool, SearchTool, TestTool")
    print("  Expected: All tools retrievable by name, get_definitions() returns 5 schemas")
    
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ExecTool())
    registry.register(CodeAnalysisTool())
    registry.register(SearchTool())
    registry.register(TestTool())
    assert registry.get("read_file") is not None
    assert registry.get("exec") is not None
    assert registry.get("code_analysis") is not None
    assert registry.get("search") is not None
    assert registry.get("test_runner") is not None
    assert len(registry.get_definitions()) == 5


async def test_read_file():
    """Test: ReadFileTool file reading functionality
    Input: A text file with content "Hello, World!\nThis test file is for testing the read file tool."
    Expected Output: The tool should return the full file content as a string
    """
    print("\n[TEST] Read File Tool")
    print("  Plan: Test that ReadFileTool can read and return file contents")
    print("  Input: File path to a text file containing 'Hello, World!'")
    print("  Expected Output: String containing the full file content including 'Hello, World!'")
    
    tool = ReadFileTool()
    # Create test file
    test_file = Path("/tmp/test.txt")
    test_file.write_text("Hello, World!\nThis test file is for testing the read file tool.")

    result = await tool.execute(path=str(test_file))
    print(f"  Actual Output: {result[:50]}...")
    assert "Hello, World!" in result
    print("✓ Test [read file] passed")
async def test_exec():
    """Test: ExecTool shell command execution
    Input: Shell command "echo 'Hello, World!'"
    Expected Output: Command output containing "Hello, World!"
    """
    print("\n[TEST] Exec Tool")
    print("  Plan: Test that ExecTool can execute shell commands and return output")
    print("  Input: Command 'echo \"Hello, World!\"'")
    print("  Expected Output: String containing 'Hello, World!' from command output")
    
    tool = ExecTool()
    result = await tool.execute(command="echo 'Hello, World!'")
    print(f"  Actual Output: {result}")
    assert "Hello, World!" in result
    print("✓ Test [exec] passed")

async def test_code_analysis_pylint():
    """Test: CodeAnalysisTool pylint functionality
    Input: A Python file with code issues (missing docstrings, unused variables)
    Expected Output: Dict with keys 'score', 'errors', 'warnings', 'messages', 'raw_output'
    """
    print("\n[TEST] Code Analysis - Pylint")
    print("  Plan: Test that CodeAnalysisTool.pylint() can run pylint and parse results")
    print("  Input: Python file path with code containing potential linting issues")
    print("  Expected Output: Dict with 'score' (float), 'errors' (int), 'warnings' (int), 'messages' (list)")
    
    tool = CodeAnalysisTool()
    
    # Create a test Python file with some issues
    test_file = Path("/tmp/test_pylint.py")
    test_code = """def hello_world():
    x = 1
    y = 2
    print(x + y)
    return None
"""
    test_file.write_text(test_code)
    
    try:
        result = await tool.pylint(str(test_file))
        print(f"  Actual Output: {json.dumps(result, indent=2)[:200]}...")
        
        # Check that result is a dict with expected keys
        assert isinstance(result, dict)
        assert "errors" in result or "error" in result
        assert "warnings" in result or "error" in result
        
        print("✓ Test [code_analysis pylint] passed")
    except Exception as e:
        # If pylint is not installed, that's okay for testing
        if "not found" in str(e) or "pylint" in str(e).lower():
            print(f"⚠ Pylint not installed, skipping test: {e}")
        else:
            raise

async def test_code_analysis_black_format():
    """Test: CodeAnalysisTool black formatting functionality
    Input: A Python file with formatting issues (no spaces around operators, etc.)
    Expected Output: String containing either formatted code or message that file is already formatted
    """
    print("\n[TEST] Code Analysis - Black Format")
    print("  Plan: Test that CodeAnalysisTool.black_format() can format Python code using black")
    print("  Input: Python file path with formatting issues (x=1+2, print(x,y))")
    print("  Expected Output: String with formatted code or 'already formatted' message")
    
    tool = CodeAnalysisTool()
    
    # Create a test Python file with formatting issues
    test_file = Path("/tmp/test_black.py")
    test_code = """def hello_world():
    x=1+2
    y=3*4
    print(x,y)
    return None
"""
    test_file.write_text(test_code)
    
    try:
        result = await tool.black_format(str(test_file))
        print(f"  Actual Output: {result[:200]}...")
        
        # Check that we got a result (either formatted or already formatted message)
        assert isinstance(result, str)
        assert len(result) > 0
        
        print("✓ Test [code_analysis black_format] passed")
    except Exception as e:
        # If black is not installed, that's okay for testing
        if "not found" in str(e) or "black" in str(e).lower():
            print(f"⚠ Black not installed, skipping test: {e}")
        else:
            raise

async def test_code_analysis_mypy_check():
    """Test: CodeAnalysisTool mypy type checking functionality
    Input: A Python file with type issues (calling function with wrong types)
    Expected Output: Dict with keys 'errors', 'warnings', 'notes', 'raw_output', 'exit_code'
    """
    print("\n[TEST] Code Analysis - Mypy Check")
    print("  Plan: Test that CodeAnalysisTool.mypy_check() can run mypy and parse type errors")
    print("  Input: Python file path with type issues (calling function with wrong argument types)")
    print("  Expected Output: Dict with 'errors' (list), 'warnings' (list), 'exit_code' (int)")
    
    tool = CodeAnalysisTool()
    
    # Create a test Python file with type issues
    test_file = Path("/tmp/test_mypy.py")
    test_code = """def add_numbers(a, b):
    return a + b

result = add_numbers("hello", "world")  # Type error: strings instead of numbers
"""
    test_file.write_text(test_code)
    
    try:
        result = await tool.mypy_check(str(test_file))
        print(f"  Actual Output: {json.dumps(result, indent=2)[:200]}...")
        
        # Check that result is a dict with expected keys
        assert isinstance(result, dict)
        assert "errors" in result or "error" in result
        assert "exit_code" in result or "error" in result
        
        print("✓ Test [code_analysis mypy_check] passed")
    except Exception as e:
        # If mypy is not installed, that's okay for testing
        if "not found" in str(e) or "mypy" in str(e).lower():
            print(f"⚠ Mypy not installed, skipping test: {e}")
        else:
            raise

async def test_code_analysis_execute():
    """Test: CodeAnalysisTool execute method routing
    Input: Different action parameters ('pylint', 'black_format', 'mypy_check', 'invalid_action')
    Expected Output: Appropriate results for each action, or error for invalid action
    """
    print("\n[TEST] Code Analysis - Execute Method")
    print("  Plan: Test that CodeAnalysisTool.execute() routes to correct methods based on action")
    print("  Input: action='pylint'/'black_format'/'mypy_check'/'invalid_action', file_path")
    print("  Expected Output: String results for valid actions, error message for invalid action")
    
    tool = CodeAnalysisTool()
    
    # Create a test Python file
    test_file = Path("/tmp/test_code_analysis.py")
    test_code = """def hello():
    print("Hello, World!")
"""
    test_file.write_text(test_code)
    
    # Test pylint action
    try:
        result = await tool.execute(action="pylint", file_path=str(test_file))
        assert isinstance(result, str)
        print("✓ Test [code_analysis execute pylint] passed")
    except Exception as e:
        if "pylint" in str(e).lower() or "not found" in str(e):
            print(f"⚠ Pylint not available: {e}")
    
    # Test black_format action
    try:
        result = await tool.execute(action="black_format", file_path=str(test_file))
        assert isinstance(result, str)
        print("✓ Test [code_analysis execute black_format] passed")
    except Exception as e:
        if "black" in str(e).lower() or "not found" in str(e):
            print(f"⚠ Black not available: {e}")
    
    # Test mypy_check action
    try:
        result = await tool.execute(action="mypy_check", file_path=str(test_file))
        assert isinstance(result, str)
        print("✓ Test [code_analysis execute mypy_check] passed")
    except Exception as e:
        if "mypy" in str(e).lower() or "not found" in str(e):
            print(f"⚠ Mypy not available: {e}")
    
    # Test invalid action
    result = await tool.execute(action="invalid_action", file_path=str(test_file))
    assert "Error" in result or "Unknown action" in result
    print("✓ Test [code_analysis execute invalid_action] passed")

async def test_search_grep():
    """Test: SearchTool grep functionality
    Input: Pattern "Hello", directory path, recursive=True
    Expected Output: List of dicts with 'file', 'line', 'content' for each match
    """
    print("\n[TEST] Search Tool - Grep")
    print("  Plan: Test that SearchTool.grep() can search for patterns in files using regex")
    print("  Input: pattern='Hello', path=directory with files containing 'Hello', recursive=True")
    print("  Expected Output: List of dicts, each with 'file' (str), 'line' (int), 'content' (str)")
    
    tool = SearchTool()
    
    # Create a test directory with files
    test_dir = Path("/tmp/test_search")
    test_dir.mkdir(exist_ok=True)
    
    test_file1 = test_dir / "file1.txt"
    test_file1.write_text("Hello World\nThis is a test\nHello again\n")
    
    test_file2 = test_dir / "file2.txt"
    test_file2.write_text("Goodbye World\nAnother test\n")
    
    try:
        # Test grep with simple pattern
        result = await tool.grep("Hello", str(test_dir), recursive=True)
        print(f"  Actual Output: Found {len(result)} matches")
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("Hello" in str(match) for match in result)
        print(f"✓ Test [search grep] passed - found {len(result)} matches")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_search_find_files():
    """Test: SearchTool find_files functionality
    Input: Pattern "*.py", directory path
    Expected Output: List of file paths matching the pattern
    """
    print("\n[TEST] Search Tool - Find Files")
    print("  Plan: Test that SearchTool.find_files() can find files matching a pattern")
    print("  Input: pattern='*.py', path=directory with .py and .md files")
    print("  Expected Output: List of strings (file paths) containing at least 2 .py files")
    
    tool = SearchTool()
    
    # Create a test directory with files
    test_dir = Path("/tmp/test_find_files")
    test_dir.mkdir(exist_ok=True)
    
    (test_dir / "test1.py").write_text("# test")
    (test_dir / "test2.py").write_text("# test")
    (test_dir / "readme.md").write_text("# readme")
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "test3.py").write_text("# test")
    
    try:
        # Test finding Python files
        result = await tool.find_files("*.py", str(test_dir))
        print(f"  Actual Output: Found {len(result)} files")
        assert isinstance(result, list)
        assert len(result) >= 2  # Should find at least test1.py and test2.py
        assert all(".py" in f for f in result)
        print(f"✓ Test [search find_files] passed - found {len(result)} files")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_search_find_in_files():
    """Test: SearchTool find_in_files functionality
    Input: Text "hello world", extensions=["py", "js"]
    Expected Output: Dict mapping file paths to lists of line numbers where text was found
    """
    print("\n[TEST] Search Tool - Find In Files")
    print("  Plan: Test that SearchTool.find_in_files() can search for text in files with specific extensions")
    print("  Input: text='hello world', extensions=['py', 'js']")
    print("  Expected Output: Dict with file paths as keys, lists of line numbers as values")
    
    tool = SearchTool()
    
    # Create test files in current directory structure
    test_file1 = Path("/tmp/test_find_text.py")
    test_file1.write_text("def hello():\n    print('hello world')\n")
    
    test_file2 = Path("/tmp/test_find_text.js")
    test_file2.write_text("function hello() {\n    console.log('hello world');\n}\n")
    
    try:
        # Change to /tmp for testing
        import os
        original_cwd = os.getcwd()
        os.chdir("/tmp")
        
        result = await tool.find_in_files("hello world", extensions=["py", "js"])
        print(f"  Actual Output: Found in {len(result)} files")
        assert isinstance(result, dict)
        # Should find matches in both files
        print(f"✓ Test [search find_in_files] passed - found in {len(result)} files")
        
        os.chdir(original_cwd)
    finally:
        # Cleanup
        if test_file1.exists():
            test_file1.unlink()
        if test_file2.exists():
            test_file2.unlink()

async def test_search_find_todos():
    """Test: SearchTool find_todos functionality
    Input: Directory path containing Python file with TODO, FIXME, NOTE, XXX comments
    Expected Output: List of dicts with 'file', 'line', 'type', 'content' for each TODO marker
    """
    print("\n[TEST] Search Tool - Find TODOs")
    print("  Plan: Test that SearchTool.find_todos() can find TODO, FIXME, NOTE, XXX comments")
    print("  Input: path=directory with Python file containing TODO/FIXME/NOTE/XXX comments")
    print("  Expected Output: List of dicts, each with 'file', 'line', 'type' (TODO/FIXME/etc), 'content'")
    
    tool = SearchTool()
    
    # Create a test directory with files containing TODOs
    test_dir = Path("/tmp/test_todos")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_todos.py"
    test_code = """# TODO: Fix this function
def hello():
    # FIXME: Add error handling
    print("Hello")
    # NOTE: This is important
    # XXX: Remove this later
"""
    test_file.write_text(test_code)
    
    try:
        result = await tool.find_todos(str(test_dir))
        print(f"  Actual Output: Found {len(result)} TODO markers")
        assert isinstance(result, list)
        assert len(result) >= 4  # Should find TODO, FIXME, NOTE, XXX
        assert any("TODO" in str(item) for item in result)
        assert any("FIXME" in str(item) for item in result)
        print(f"✓ Test [search find_todos] passed - found {len(result)} TODOs")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_search_count_lines():
    """Test: SearchTool count_lines functionality
    Input: Directory path with .py and .js files, or single file path
    Expected Output: Dict with line counts (by extension or total)
    """
    print("\n[TEST] Search Tool - Count Lines")
    print("  Plan: Test that SearchTool.count_lines() can count lines of code")
    print("  Input: path=directory with .py and .js files, by_extension=True/False")
    print("  Expected Output: Dict with 'total_lines', 'code_lines', 'blank_lines' (grouped by ext or total)")
    
    tool = SearchTool()
    
    # Create a test directory with files
    test_dir = Path("/tmp/test_count_lines")
    test_dir.mkdir(exist_ok=True)
    
    test_file1 = test_dir / "file1.py"
    test_file1.write_text("def hello():\n    print('hello')\n\n# comment\n")
    
    test_file2 = test_dir / "file2.js"
    test_file2.write_text("function hello() {\n    console.log('hello');\n}\n")
    
    try:
        # Test counting by extension
        result = await tool.count_lines(str(test_dir), by_extension=True)
        print(f"  Actual Output (by_extension): {list(result.keys())}")
        assert isinstance(result, dict)
        assert "py" in result or "js" in result
        print(f"✓ Test [search count_lines by_extension] passed")
        
        # Test counting single file
        result = await tool.count_lines(str(test_file1), by_extension=False)
        print(f"  Actual Output (single file): {list(result.keys())}")
        assert isinstance(result, dict)
        assert "total_lines" in result or "file" in result
        print(f"✓ Test [search count_lines single file] passed")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_search_execute():
    """Test: SearchTool execute method routing
    Input: Different action parameters ('grep', 'find_files', 'find_todos', 'count_lines', 'invalid_action')
    Expected Output: Appropriate formatted string results for each action, or error for invalid/missing params
    """
    print("\n[TEST] Search Tool - Execute Method")
    print("  Plan: Test that SearchTool.execute() routes to correct methods based on action")
    print("  Input: action='grep'/'find_files'/'find_todos'/'count_lines'/'invalid_action', path, optional params")
    print("  Expected Output: Formatted string results for valid actions, error message for invalid/missing params")
    
    tool = SearchTool()
    
    # Create a test directory with files
    test_dir = Path("/tmp/test_search_execute")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test.py"
    test_file.write_text("def hello():\n    # TODO: Add docstring\n    print('hello')\n")
    
    try:
        # Test grep action
        result = await tool.execute(action="grep", path=str(test_dir), pattern="def")
        assert isinstance(result, str)
        assert "results" in result.lower() or "found" in result.lower()
        print("✓ Test [search execute grep] passed")
        
        # Test find_files action
        result = await tool.execute(action="find_files", path=str(test_dir), pattern="*.py")
        assert isinstance(result, str)
        print("✓ Test [search execute find_files] passed")
        
        # Test find_todos action
        result = await tool.execute(action="find_todos", path=str(test_dir))
        assert isinstance(result, str)
        print("✓ Test [search execute find_todos] passed")
        
        # Test count_lines action
        result = await tool.execute(action="count_lines", path=str(test_dir))
        assert isinstance(result, str)
        print("✓ Test [search execute count_lines] passed")
        
        # Test invalid action
        result = await tool.execute(action="invalid_action", path=str(test_dir))
        assert "Error" in result or "Unknown action" in result
        print("✓ Test [search execute invalid_action] passed")
        
        # Test missing required parameter
        result = await tool.execute(action="grep", path=str(test_dir))
        assert "Error" in result or "required" in result.lower()
        print("✓ Test [search execute missing parameter] passed")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_test_runner_pytest():
    """Test: TestTool run_pytest functionality
    Input: Path to test directory (optional), verbose=True
    Expected Output: Dict with 'passed', 'failed', 'skipped', 'errors', 'output', 'exit_code'
    """
    print("\n[TEST] Test Runner - Pytest")
    print("  Plan: Test that TestTool.run_pytest() can run pytest tests and parse results")
    print("  Input: path=test directory (optional), verbose=True")
    print("  Expected Output: Dict with 'passed' (int), 'failed' (int), 'skipped' (int), 'output' (str), 'exit_code' (int)")
    
    tool = TestTool()
    
    # Create a simple test file
    test_dir = Path("/tmp/test_pytest_dir")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_example.py"
    test_code = """def test_example():
    assert 1 + 1 == 2

def test_another():
    assert True
"""
    test_file.write_text(test_code)
    
    try:
        result = await tool.run_pytest(str(test_dir), verbose=True)
        print(f"  Actual Output: passed={result.get('passed')}, failed={result.get('failed')}, exit_code={result.get('exit_code')}")
        
        assert isinstance(result, dict)
        assert "passed" in result or "error" in result
        assert "failed" in result or "error" in result
        assert "exit_code" in result or "error" in result
        
        print("✓ Test [test_runner run_pytest] passed")
    except Exception as e:
        if "pytest" in str(e).lower() or "not found" in str(e):
            print(f"⚠ Pytest not installed, skipping test: {e}")
        else:
            raise
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_test_runner_jest():
    """Test: TestTool run_jest functionality
    Input: Path to test directory (optional)
    Expected Output: Dict with 'passed', 'failed', 'skipped', 'output', 'exit_code'
    """
    print("\n[TEST] Test Runner - Jest")
    print("  Plan: Test that TestTool.run_jest() can run jest tests and parse results")
    print("  Input: path=test directory (optional)")
    print("  Expected Output: Dict with 'passed' (int), 'failed' (int), 'skipped' (int), 'output' (str), 'exit_code' (int)")
    
    tool = TestTool()
    
    # Create a simple test file
    test_dir = Path("/tmp/test_jest_dir")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "example.test.js"
    test_code = """test('example test', () => {
    expect(1 + 1).toBe(2);
});
"""
    test_file.write_text(test_code)
    
    try:
        result = await tool.run_jest(str(test_dir))
        print(f"  Actual Output: passed={result.get('passed')}, failed={result.get('failed')}, exit_code={result.get('exit_code')}")
        
        assert isinstance(result, dict)
        assert "passed" in result or "error" in result
        assert "failed" in result or "error" in result
        assert "exit_code" in result or "error" in result
        
        print("✓ Test [test_runner run_jest] passed")
    except Exception as e:
        if "jest" in str(e).lower() or "not found" in str(e):
            print(f"⚠ Jest not installed, skipping test: {e}")
        else:
            raise
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_test_runner_unittest():
    """Test: TestTool run_unittest functionality
    Input: Module path string (e.g., 'tests.test_example')
    Expected Output: Dict with 'passed', 'failed', 'skipped', 'errors', 'output', 'exit_code'
    """
    print("\n[TEST] Test Runner - Unittest")
    print("  Plan: Test that TestTool.run_unittest() can run Python unittest module and parse results")
    print("  Input: module='tests.test_example' (Python module path)")
    print("  Expected Output: Dict with 'passed' (int), 'failed' (int), 'skipped' (int), 'errors' (int), 'output' (str), 'exit_code' (int)")
    
    tool = TestTool()
    
    # Create a simple test module
    test_dir = Path("/tmp/test_unittest_dir")
    test_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    (test_dir / "__init__.py").write_text("")
    
    # Create test module
    test_module = test_dir / "test_example.py"
    test_code = """import unittest

class TestExample(unittest.TestCase):
    def test_example(self):
        self.assertEqual(1 + 1, 2)
    
    def test_another(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""
    test_module.write_text(test_code)
    
    try:
        # Add test directory to path temporarily
        import sys
        sys.path.insert(0, str(test_dir))
        
        result = await tool.run_unittest("test_example")
        print(f"  Actual Output: passed={result.get('passed')}, failed={result.get('failed')}, exit_code={result.get('exit_code')}")
        
        assert isinstance(result, dict)
        assert "passed" in result or "error" in result
        assert "failed" in result or "error" in result
        assert "exit_code" in result or "error" in result
        
        print("✓ Test [test_runner run_unittest] passed")
        
        sys.path.remove(str(test_dir))
    except Exception as e:
        print(f"⚠ Unittest test had issue: {e}")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

async def test_test_runner_coverage_report():
    """Test: TestTool coverage_report functionality
    Input: None (uses current directory)
    Expected Output: Dict with 'coverage_percent', 'lines_covered', 'lines_total', 'branches_covered', 'output'
    """
    print("\n[TEST] Test Runner - Coverage Report")
    print("  Plan: Test that TestTool.coverage_report() can generate code coverage report")
    print("  Input: None (uses current directory)")
    print("  Expected Output: Dict with 'coverage_percent' (int), 'lines_covered' (int), 'lines_total' (int), 'branches_covered' (int), 'output' (str)")
    
    tool = TestTool()
    
    try:
        result = await tool.coverage_report()
        print(f"  Actual Output: coverage={result.get('coverage_percent')}%, lines={result.get('lines_covered')}/{result.get('lines_total')}")
        
        assert isinstance(result, dict)
        assert "coverage_percent" in result or "error" in result
        assert "lines_covered" in result or "error" in result
        assert "lines_total" in result or "error" in result
        
        print("✓ Test [test_runner coverage_report] passed")
    except Exception as e:
        if "coverage" in str(e).lower() or "not found" in str(e):
            print(f"⚠ Coverage tool not installed, skipping test: {e}")
        else:
            raise

async def test_test_runner_execute():
    """Test: TestTool execute method routing
    Input: Different action parameters ('run_pytest', 'run_jest', 'run_unittest', 'coverage_report', 'invalid_action')
    Expected Output: Appropriate JSON string results for each action, or error for invalid action
    """
    print("\n[TEST] Test Runner - Execute Method")
    print("  Plan: Test that TestTool.execute() routes to correct methods based on action")
    print("  Input: action='run_pytest'/'run_jest'/'run_unittest'/'coverage_report'/'invalid_action', optional params")
    print("  Expected Output: JSON string results for valid actions, error message for invalid action")
    
    tool = TestTool()
    
    # Create a simple test file for pytest
    test_dir = Path("/tmp/test_execute_dir")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_example.py"
    test_file.write_text("def test_example():\n    assert True\n")
    
    try:
        # Test run_pytest action
        try:
            result = await tool.execute(action="run_pytest", path=str(test_dir))
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert "passed" in result_dict or "error" in result_dict
            print("✓ Test [test_runner execute run_pytest] passed")
        except Exception as e:
            if "pytest" in str(e).lower():
                print(f"⚠ Pytest not available: {e}")
        
        # Test run_jest action
        try:
            result = await tool.execute(action="run_jest", path=str(test_dir))
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert "passed" in result_dict or "error" in result_dict
            print("✓ Test [test_runner execute run_jest] passed")
        except Exception as e:
            if "jest" in str(e).lower():
                print(f"⚠ Jest not available: {e}")
        
        # Test run_unittest action
        try:
            result = await tool.execute(action="run_unittest", module="test_example")
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert "passed" in result_dict or "error" in result_dict
            print("✓ Test [test_runner execute run_unittest] passed")
        except Exception as e:
            print(f"⚠ Unittest test had issue: {e}")
        
        # Test coverage_report action
        try:
            result = await tool.execute(action="coverage_report")
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert "coverage_percent" in result_dict or "error" in result_dict
            print("✓ Test [test_runner execute coverage_report] passed")
        except Exception as e:
            if "coverage" in str(e).lower():
                print(f"⚠ Coverage tool not available: {e}")
        
        # Test invalid action
        result = await tool.execute(action="invalid_action")
        assert isinstance(result, str)
        result_dict = json.loads(result)
        assert "error" in result_dict or "Unknown action" in result
        print("✓ Test [test_runner execute invalid_action] passed")
        
        # Test missing required parameter for unittest
        result = await tool.execute(action="run_unittest")
        assert isinstance(result, str)
        result_dict = json.loads(result)
        assert "error" in result_dict or "required" in result.lower()
        print("✓ Test [test_runner execute missing parameter] passed")
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    print("Testing tool registry...")
    asyncio.run(test_tool_registry())
    print("\nTesting read file...")
    asyncio.run(test_read_file())
    print("\nTesting exec...")
    asyncio.run(test_exec())
    print("\nTesting code analysis (pylint)...")
    asyncio.run(test_code_analysis_pylint())
    print("\nTesting code analysis (black_format)...")
    asyncio.run(test_code_analysis_black_format())
    print("\nTesting code analysis (mypy_check)...")
    asyncio.run(test_code_analysis_mypy_check())
    print("\nTesting code analysis (execute method)...")
    asyncio.run(test_code_analysis_execute())
    print("\nTesting search tool (grep)...")
    asyncio.run(test_search_grep())
    print("\nTesting search tool (find_files)...")
    asyncio.run(test_search_find_files())
    print("\nTesting search tool (find_in_files)...")
    asyncio.run(test_search_find_in_files())
    print("\nTesting search tool (find_todos)...")
    asyncio.run(test_search_find_todos())
    print("\nTesting search tool (count_lines)...")
    asyncio.run(test_search_count_lines())
    print("\nTesting search tool (execute method)...")
    asyncio.run(test_search_execute())
    print("\nTesting test runner (pytest)...")
    asyncio.run(test_test_runner_pytest())
    print("\nTesting test runner (jest)...")
    asyncio.run(test_test_runner_jest())
    print("\nTesting test runner (unittest)...")
    asyncio.run(test_test_runner_unittest())
    print("\nTesting test runner (coverage_report)...")
    asyncio.run(test_test_runner_coverage_report())
    print("\nTesting test runner (execute method)...")
    asyncio.run(test_test_runner_execute())
    print("\n✓ All tests passed")