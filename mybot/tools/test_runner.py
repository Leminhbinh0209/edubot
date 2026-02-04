"""Test runner tools for pytest, jest, unittest, and coverage."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from mybot.tools.base import Tool


class TestTool(Tool):
    """Tool for running tests using pytest, jest, unittest, and generating coverage reports."""
    
    def __init__(self, timeout: int = 300):  # 5 minutes default for tests
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "test_runner"
    
    @property
    def description(self) -> str:
        return """Run tests and generate coverage reports. Supports:
        - run_pytest: Run pytest tests (Python)
        - run_jest: Run jest tests (JavaScript/TypeScript)
        - run_unittest: Run Python unittest module
        - coverage_report: Generate code coverage report
        
        Use the 'action' parameter to specify which test runner to use."""
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run_pytest", "run_jest", "run_unittest", "coverage_report"],
                    "description": "Which test action to perform"
                },
                "path": {
                    "type": "string",
                    "description": "Path to test directory/file (for pytest, jest, coverage)"
                },
                "module": {
                    "type": "string",
                    "description": "Python module path for unittest (e.g., 'tests.test_example')"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Verbose output (default: true for pytest)",
                    "default": True
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        action: str,
        path: str | None = None,
        module: str | None = None,
        verbose: bool = True,
        **kwargs: Any
    ) -> str:
        """Execute the specified test action."""
        try:
            if action == "run_pytest":
                result = await self.run_pytest(path, verbose)
                return json.dumps(result, indent=2)
            
            elif action == "run_jest":
                result = await self.run_jest(path)
                return json.dumps(result, indent=2)
            
            elif action == "run_unittest":
                if not module:
                    return json.dumps({"error": "Module parameter is required for unittest"}, indent=2)
                result = await self.run_unittest(module)
                return json.dumps(result, indent=2)
            
            elif action == "coverage_report":
                result = await self.coverage_report()
                return json.dumps(result, indent=2)
            
            else:
                return json.dumps({"error": f"Unknown action '{action}'"}, indent=2)
        
        except Exception as e:
            return json.dumps({"error": f"Error running {action}: {str(e)}"}, indent=2)
    
    async def run_pytest(self, path: str | None = None, verbose: bool = True) -> dict:
        """Run pytest tests.
        
        Args:
            path: Path to test directory or file (default: current directory)
            verbose: Whether to show verbose output
        
        Returns:
            Dict with keys: 'passed', 'failed', 'skipped', 'errors', 'output', 'exit_code'
        """
        try:
            # Check if pytest is available
            check_process = await asyncio.create_subprocess_exec(
                "pytest",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                return {
                    "error": "pytest not found. Install it with: pip install pytest",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "output": "",
                    "exit_code": -1
                }
            
            # Build pytest command
            cmd = ["pytest"]
            if verbose:
                cmd.append("-v")
            else:
                cmd.append("-q")
            
            # Add path if provided
            if path:
                path_obj = Path(path)
                if path_obj.exists():
                    cmd.append(str(path_obj))
                else:
                    return {
                        "error": f"Path not found: {path}",
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "errors": 0,
                        "output": "",
                        "exit_code": -1
                    }
            
            # Run pytest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "error": f"pytest timed out after {self.timeout} seconds",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "output": "",
                    "exit_code": -1
                }
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            full_output = output + "\n" + error_output if error_output else output
            
            # Parse pytest output
            passed = 0
            failed = 0
            skipped = 0
            errors = 0
            
            # Look for summary line: "X passed, Y failed, Z skipped" or similar
            summary_pattern = re.compile(
                r'(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped|(\d+)\s+error',
                re.IGNORECASE
            )
            
            # Search in reverse (summary is usually at the end)
            for line in reversed(output.split('\n')):
                matches = summary_pattern.findall(line)
                if matches:
                    for match in matches:
                        if match[0]:  # passed
                            passed = int(match[0])
                        elif match[1]:  # failed
                            failed = int(match[1])
                        elif match[2]:  # skipped
                            skipped = int(match[2])
                        elif match[3]:  # error
                            errors = int(match[3])
                    break
            
            # Also try to find in the last few lines
            if passed == 0 and failed == 0:
                last_lines = output.split('\n')[-10:]
                for line in last_lines:
                    if 'passed' in line.lower() and 'failed' in line.lower():
                        # Try to extract numbers
                        nums = re.findall(r'\d+', line)
                        if len(nums) >= 2:
                            passed = int(nums[0])
                            failed = int(nums[1]) if len(nums) > 1 else 0
            
            return {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "output": full_output[:5000],  # Limit output size
                "exit_code": process.returncode
            }
            
        except FileNotFoundError:
            return {
                "error": "pytest not found. Install it with: pip install pytest",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "output": "",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "error": f"Error running pytest: {str(e)}",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "output": "",
                "exit_code": -1
            }
    
    async def run_jest(self, path: str | None = None) -> dict:
        """Run jest tests.
        
        Args:
            path: Path to test directory or file (default: current directory)
        
        Returns:
            Dict with keys: 'passed', 'failed', 'skipped', 'output', 'exit_code'
        """
        try:
            # Check if jest is available
            check_process = await asyncio.create_subprocess_exec(
                "jest",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                return {
                    "error": "jest not found. Install it with: npm install --save-dev jest",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "output": "",
                    "exit_code": -1
                }
            
            # Build jest command
            cmd = ["jest"]
            if path:
                path_obj = Path(path)
                if path_obj.exists():
                    cmd.append(str(path_obj))
                else:
                    return {
                        "error": f"Path not found: {path}",
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "output": "",
                        "exit_code": -1
                    }
            
            # Run jest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "error": f"jest timed out after {self.timeout} seconds",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "output": "",
                    "exit_code": -1
                }
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            full_output = output + "\n" + error_output if error_output else output
            
            # Parse jest output
            passed = 0
            failed = 0
            skipped = 0
            
            # Jest output format: "Tests: X passed, Y failed, Z total"
            summary_pattern = re.compile(
                r'Tests:\s+(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped',
                re.IGNORECASE
            )
            
            # Also try simpler patterns
            passed_match = re.search(r'(\d+)\s+passed', output, re.IGNORECASE)
            failed_match = re.search(r'(\d+)\s+failed', output, re.IGNORECASE)
            skipped_match = re.search(r'(\d+)\s+skipped', output, re.IGNORECASE)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            if skipped_match:
                skipped = int(skipped_match.group(1))
            
            return {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "output": full_output[:5000],  # Limit output size
                "exit_code": process.returncode
            }
            
        except FileNotFoundError:
            return {
                "error": "jest not found. Install it with: npm install --save-dev jest",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "output": "",
                "exit_code": -1
            }
        except Exception as e:
            return {
                "error": f"Error running jest: {str(e)}",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "output": "",
                "exit_code": -1
            }
    
    async def run_unittest(self, module: str) -> dict:
        """Run Python unittest module.
        
        Args:
            module: Python module path (e.g., 'tests.test_example' or 'tests.test_example.TestClass')
        
        Returns:
            Dict with keys: 'passed', 'failed', 'skipped', 'errors', 'output', 'exit_code'
        """
        try:
            # Build unittest command
            cmd = ["python", "-m", "unittest", module]
            
            # Run unittest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "error": f"unittest timed out after {self.timeout} seconds",
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "output": "",
                    "exit_code": -1
                }
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            full_output = output + "\n" + error_output if error_output else output
            
            # Parse unittest output
            passed = 0
            failed = 0
            skipped = 0
            errors = 0
            total = 0
            
            # unittest output format: "Ran X tests in Y seconds" and "OK" or "FAILED"
            ran_match = re.search(r'Ran\s+(\d+)\s+test', output, re.IGNORECASE)
            if ran_match:
                total = int(ran_match.group(1))
            
            # Check for OK or FAILED
            if "OK" in output:
                passed = total if ran_match else 0
            elif "FAILED" in output or "ERROR" in output:
                # Try to extract failure count
                failed_match = re.search(r'failures?[=:]?\s*(\d+)', output, re.IGNORECASE)
                error_match = re.search(r'errors?[=:]?\s*(\d+)', output, re.IGNORECASE)
                
                if failed_match:
                    failed = int(failed_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))
                
                # If we have total but no specific counts, estimate
                if ran_match and failed == 0 and errors == 0:
                    failed = total
            
            return {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "errors": errors,
                "output": full_output[:5000],  # Limit output size
                "exit_code": process.returncode
            }
            
        except Exception as e:
            return {
                "error": f"Error running unittest: {str(e)}",
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "output": "",
                "exit_code": -1
            }
    
    async def coverage_report(self) -> dict:
        """Generate code coverage report.
        
        Returns:
            Dict with keys: 'coverage_percent', 'lines_covered', 'lines_total', 'branches_covered', 'output'
        """
        try:
            # Check if coverage is available (pytest-cov or coverage.py)
            # Try pytest-cov first
            check_process = await asyncio.create_subprocess_exec(
                "pytest",
                "--cov",
                "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await check_process.communicate()
            
            use_pytest_cov = check_process.returncode == 0
            
            if not use_pytest_cov:
                # Try coverage.py
                check_process = await asyncio.create_subprocess_exec(
                    "coverage",
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await check_process.communicate()
                
                if check_process.returncode != 0:
                    return {
                        "error": "Coverage tool not found. Install with: pip install pytest-cov or pip install coverage",
                        "coverage_percent": 0,
                        "lines_covered": 0,
                        "lines_total": 0,
                        "branches_covered": 0,
                        "output": ""
                    }
            
            if use_pytest_cov:
                # Use pytest with coverage
                process = await asyncio.create_subprocess_exec(
                    "pytest",
                    "--cov",
                    "--cov-report=term",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                # Use coverage.py
                process = await asyncio.create_subprocess_exec(
                    "coverage",
                    "run",
                    "-m",
                    "pytest",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "error": f"Coverage report timed out after {self.timeout} seconds",
                    "coverage_percent": 0,
                    "lines_covered": 0,
                    "lines_total": 0,
                    "branches_covered": 0,
                    "output": ""
                }
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            full_output = output + "\n" + error_output if error_output else output
            
            # Parse coverage output
            coverage_percent = 0
            lines_covered = 0
            lines_total = 0
            branches_covered = 0
            
            # Look for coverage percentage: "TOTAL X% coverage" or "X%"
            coverage_match = re.search(r'TOTAL\s+(\d+)%|(\d+)%\s+coverage', output, re.IGNORECASE)
            if not coverage_match:
                coverage_match = re.search(r'(\d+)%', output)
            
            if coverage_match:
                coverage_percent = int(coverage_match.group(1) or coverage_match.group(2))
            
            # Look for line counts: "X/Y lines covered"
            lines_match = re.search(r'(\d+)/(\d+)\s+lines', output, re.IGNORECASE)
            if lines_match:
                lines_covered = int(lines_match.group(1))
                lines_total = int(lines_match.group(2))
            
            # Look for branch coverage
            branches_match = re.search(r'(\d+)/(\d+)\s+branches', output, re.IGNORECASE)
            if branches_match:
                branches_covered = int(branches_match.group(1))
            
            return {
                "coverage_percent": coverage_percent,
                "lines_covered": lines_covered,
                "lines_total": lines_total,
                "branches_covered": branches_covered,
                "output": full_output[:5000]  # Limit output size
            }
            
        except Exception as e:
            return {
                "error": f"Error generating coverage report: {str(e)}",
                "coverage_percent": 0,
                "lines_covered": 0,
                "lines_total": 0,
                "branches_covered": 0,
                "output": ""
            }
