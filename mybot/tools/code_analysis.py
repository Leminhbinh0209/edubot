"""Code analysis tools for linting, formatting, and type checking."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any
from mybot.tools.base import Tool


class CodeAnalysisTool(Tool):
    """Tool for running linters, formatters, and type checkers on Python code."""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "code_analysis"
    
    @property
    def description(self) -> str:
        return """Run code analysis tools on Python files. Supports:
        - pylint: Run pylint linter and get analysis results
        - black_format: Auto-format Python code using black
        - mypy_check: Run mypy type checker on Python code
        
        Use the 'action' parameter to specify which tool to run."""
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pylint", "black_format", "mypy_check"],
                    "description": "Which analysis tool to run: 'pylint', 'black_format', or 'mypy_check'"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the Python file to analyze"
                }
            },
            "required": ["action", "file_path"]
        }
    
    async def execute(self, action: str, file_path: str, **kwargs: Any) -> str:
        """Execute the specified code analysis action."""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            return f"Error: File not found: {file_path}"
        
        if not file_path_obj.suffix == ".py":
            return f"Error: File must be a Python file (.py), got: {file_path_obj.suffix}"
        
        try:
            if action == "pylint":
                result = await self.pylint(file_path)
                return json.dumps(result, indent=2)
            elif action == "black_format":
                result = await self.black_format(file_path)
                return result
            elif action == "mypy_check":
                result = await self.mypy_check(file_path)
                return json.dumps(result, indent=2)
            else:
                return f"Error: Unknown action '{action}'. Must be one of: pylint, black_format, mypy_check"
        except Exception as e:
            return f"Error running {action}: {str(e)}"
    
    async def pylint(self, file_path: str) -> dict:
        """Run pylint and parse output.
        
        Returns:
            dict with keys: 'score', 'errors', 'warnings', 'messages', 'raw_output'
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "pylint",
                file_path,
                "--output-format=json",
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
                    "error": f"pylint timed out after {self.timeout} seconds",
                    "score": None,
                    "errors": 0,
                    "warnings": 0,
                    "messages": []
                }
            
            raw_output = stdout.decode("utf-8", errors="replace")
            stderr_output = stderr.decode("utf-8", errors="replace")
            
            # Parse JSON output
            messages = []
            if raw_output.strip():
                try:
                    messages = json.loads(raw_output)
                except json.JSONDecodeError:
                    # Fallback: try to parse text output
                    pass
            
            # Count errors and warnings
            errors = sum(1 for msg in messages if msg.get("type") == "error")
            warnings = sum(1 for msg in messages if msg.get("type") in ["warning", "convention", "refactor"])
            
            # Extract score from stderr (pylint prints score to stderr)
            score = None
            score_match = re.search(r"rated at ([\d.]+)/10", stderr_output)
            if score_match:
                score = float(score_match.group(1))
            
            return {
                "score": score,
                "errors": errors,
                "warnings": warnings,
                "messages": messages[:50],  # Limit to first 50 messages
                "raw_output": raw_output[:2000] if len(raw_output) > 2000 else raw_output
            }
            
        except FileNotFoundError:
            return {
                "error": "pylint not found. Install it with: pip install pylint",
                "score": None,
                "errors": 0,
                "warnings": 0,
                "messages": []
            }
        except Exception as e:
            return {
                "error": f"Error running pylint: {str(e)}",
                "score": None,
                "errors": 0,
                "warnings": 0,
                "messages": []
            }
    
    async def black_format(self, file_path: str) -> str:
        """Auto-format Python code using black.
        
        Returns:
            Formatted code as string, or error message
        """
        try:
            # First, check if black is available
            check_process = await asyncio.create_subprocess_exec(
                "black",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                return "Error: black formatter not found. Install it with: pip install black"
            
            # Read the original file
            file_path_obj = Path(file_path)
            original_content = file_path_obj.read_text(encoding="utf-8")
            
            # Run black in check mode first to see if formatting is needed
            check_process = await asyncio.create_subprocess_exec(
                "black",
                "--check",
                "--diff",
                file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    check_process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                check_process.kill()
                return "Error: black formatter timed out"
            
            # If file needs formatting, run black to format it
            if check_process.returncode != 0:
                format_process = await asyncio.create_subprocess_exec(
                    "black",
                    file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        format_process.communicate(),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    format_process.kill()
                    return "Error: black formatter timed out"
                
                if format_process.returncode == 0:
                    # Read the formatted file
                    formatted_content = file_path_obj.read_text(encoding="utf-8")
                    return f"File formatted successfully.\n\nFormatted content:\n{formatted_content}"
                else:
                    error_msg = stderr.decode("utf-8", errors="replace")
                    return f"Error formatting file: {error_msg}"
            else:
                # File is already formatted
                return "File is already properly formatted according to black."
                
        except FileNotFoundError:
            return "Error: black formatter not found. Install it with: pip install black"
        except Exception as e:
            return f"Error running black: {str(e)}"
    
    async def mypy_check(self, file_path: str) -> dict:
        """Type checking using mypy.
        
        Returns:
            dict with keys: 'errors', 'warnings', 'notes', 'raw_output', 'exit_code'
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mypy",
                file_path,
                "--show-error-codes",
                "--no-error-summary",
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
                    "error": f"mypy timed out after {self.timeout} seconds",
                    "errors": [],
                    "warnings": [],
                    "notes": [],
                    "exit_code": -1
                }
            
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")
            
            # Parse mypy output
            errors = []
            warnings = []
            notes = []
            
            # Mypy output format: file:line: error: message [error-code]
            error_pattern = re.compile(r'^(.+):(\d+):\s*(error|warning|note):\s*(.+?)(?:\s+\[(.+)\])?$', re.MULTILINE)
            
            for line in output.split('\n'):
                if not line.strip():
                    continue
                
                match = error_pattern.match(line)
                if match:
                    file_name, line_num, msg_type, message, error_code = match.groups()
                    entry = {
                        "file": file_name,
                        "line": int(line_num),
                        "message": message,
                        "code": error_code
                    }
                    
                    if msg_type == "error":
                        errors.append(entry)
                    elif msg_type == "warning":
                        warnings.append(entry)
                    elif msg_type == "note":
                        notes.append(entry)
            
            return {
                "errors": errors[:50],  # Limit to first 50 errors
                "warnings": warnings[:50],
                "notes": notes[:20],
                "raw_output": output[:2000] if len(output) > 2000 else output,
                "exit_code": process.returncode
            }
            
        except FileNotFoundError:
            return {
                "error": "mypy not found. Install it with: pip install mypy",
                "errors": [],
                "warnings": [],
                "notes": [],
                "exit_code": -1
            }
        except Exception as e:
            return {
                "error": f"Error running mypy: {str(e)}",
                "errors": [],
                "warnings": [],
                "notes": [],
                "exit_code": -1
            }
