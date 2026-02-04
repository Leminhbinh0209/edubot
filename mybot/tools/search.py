"""Advanced file search and analysis tools."""

import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from mybot.tools.base import Tool


class SearchTool(Tool):
    """Advanced tool for searching files, patterns, and code analysis.
    
    Better than basic file read - provides grep, file finding, text search,
    TODO detection, and line counting capabilities.
    """
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def description(self) -> str:
        return """Advanced file search and analysis tool. USE THIS TOOL when user asks to:
        - Search for patterns/text in files (action="grep" or "find_in_files")
        - Find files by name/pattern (action="find_files")
        - Find TODO/FIXME comments (action="find_todos")
        - Count lines of code (action="count_lines")
        
        Always use this tool for file searching, pattern matching, or code analysis requests.
        Required parameters: action and path."""
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["grep", "find_files", "find_in_files", "find_todos", "count_lines"],
                    "description": "Which search operation to perform"
                },
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (for grep, find_files)"
                },
                "path": {
                    "type": "string",
                    "description": "Path to search in (directory or file)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively (default: true for grep, false for find_files)",
                    "default": True
                },
                "text": {
                    "type": "string",
                    "description": "Text to search for (for find_in_files)"
                },
                "extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to search in (e.g., ['py', 'js', 'ts'])"
                }
            },
            "required": ["action", "path"]
        }
    
    async def execute(
        self,
        action: str,
        path: str,
        pattern: str | None = None,
        recursive: bool = True,
        text: str | None = None,
        extensions: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        """Execute the specified search action."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            return f"Error: Path not found: {path}"
        
        try:
            if action == "grep":
                if not pattern:
                    return "Error: 'pattern' parameter is required for grep action"
                result = await self.grep(pattern, path, recursive)
                return self._format_list_result(result, "grep")
            
            elif action == "find_files":
                if not pattern:
                    return "Error: 'pattern' parameter is required for find_files action"
                result = await self.find_files(pattern, path)
                return self._format_list_result(result, "find_files")
            
            elif action == "find_in_files":
                if not text:
                    return "Error: 'text' parameter is required for find_in_files action"
                result = await self.find_in_files(text, extensions)
                return self._format_dict_result(result, "find_in_files")
            
            elif action == "find_todos":
                result = await self.find_todos(path)
                return self._format_list_result(result, "find_todos")
            
            elif action == "count_lines":
                result = await self.count_lines(path, by_extension=True)
                return self._format_dict_result(result, "count_lines")
            
            else:
                return f"Error: Unknown action '{action}'. Must be one of: grep, find_files, find_in_files, find_todos, count_lines"
        
        except Exception as e:
            return f"Error running {action}: {str(e)}"
    
    def _format_list_result(self, result: list, action_name: str) -> str:
        """Format a list result for display."""
        if not result:
            return f"No results found for {action_name}."
        
        if len(result) > 100:
            preview = result[:100]
            return f"Found {len(result)} results (showing first 100):\n" + "\n".join(str(item) for item in preview)
        else:
            return f"Found {len(result)} results:\n" + "\n".join(str(item) for item in result)
    
    def _format_dict_result(self, result: dict, action_name: str) -> str:
        """Format a dict result for display."""
        if not result:
            return f"No results found for {action_name}."
        
        lines = [f"{action_name} results:"]
        for key, value in result.items():
            if isinstance(value, (list, dict)):
                lines.append(f"  {key}: {len(value)} items")
            else:
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    async def grep(self, pattern: str, path: str, recursive: bool = True) -> list:
        """Search for patterns in files using regex.
        
        Args:
            pattern: Regex pattern to search for
            path: Directory or file path to search in
            recursive: Whether to search recursively (default: True)
        
        Returns:
            List of matches, each as a dict with 'file', 'line', 'content'
        """
        path_obj = Path(path)
        matches = []
        
        try:
            # Compile regex pattern
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [f"Error: Invalid regex pattern: {str(e)}"]
        
        # Determine files to search
        files_to_search = []
        if path_obj.is_file():
            files_to_search = [path_obj]
        elif path_obj.is_dir():
            if recursive:
                files_to_search = list(path_obj.rglob("*"))
            else:
                files_to_search = list(path_obj.glob("*"))
            # Filter to only regular files (not directories)
            files_to_search = [f for f in files_to_search if f.is_file()]
        else:
            return [f"Error: Path is not a file or directory: {path}"]
        
        # Search in each file
        for file_path in files_to_search:
            try:
                # Skip binary files
                if self._is_binary_file(file_path):
                    continue
                
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append({
                            "file": str(file_path.relative_to(path_obj.parent) if path_obj.is_dir() else file_path),
                            "line": line_num,
                            "content": line.strip()[:200]  # Limit line length
                        })
            
            except (UnicodeDecodeError, PermissionError, IOError):
                # Skip files we can't read
                continue
        
        return matches
    
    async def find_files(self, pattern: str, path: str) -> list:
        """Find files matching a pattern.
        
        Args:
            pattern: Filename pattern (supports wildcards like *.py, test_*.py)
            path: Directory to search in
        
        Returns:
            List of matching file paths
        """
        path_obj = Path(path)
        
        if not path_obj.is_dir():
            return [f"Error: Path must be a directory: {path}"]
        
        matches = []
        
        # Convert pattern to glob pattern
        # Support both simple patterns and regex-like patterns
        if '*' in pattern or '?' in pattern:
            # Use glob pattern
            for file_path in path_obj.rglob(pattern):
                if file_path.is_file():
                    matches.append(str(file_path.relative_to(path_obj)))
        else:
            # Simple name search - search in filename
            pattern_lower = pattern.lower()
            for file_path in path_obj.rglob("*"):
                if file_path.is_file() and pattern_lower in file_path.name.lower():
                    matches.append(str(file_path.relative_to(path_obj)))
        
        return sorted(matches)
    
    async def find_in_files(self, text: str, extensions: list[str] | None = None) -> dict:
        """Search for text in files with specific extensions.
        
        Args:
            text: Text to search for
            extensions: List of file extensions to search in (e.g., ['py', 'js'])
                        If None, searches in common text files
        
        Returns:
            Dict mapping file paths to list of line numbers where text was found
        """
        if extensions is None:
            extensions = ['py', 'js', 'ts', 'jsx', 'tsx', 'md', 'txt', 'json', 'yaml', 'yml']
        
        # Normalize extensions (remove dots if present)
        extensions = [ext.lstrip('.') for ext in extensions]
        
        results = {}
        text_lower = text.lower()
        
        # Search in current directory and subdirectories
        current_dir = Path.cwd()
        
        for ext in extensions:
            pattern = f"*.{ext}"
            for file_path in current_dir.rglob(pattern):
                if not file_path.is_file():
                    continue
                
                try:
                    if self._is_binary_file(file_path):
                        continue
                    
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.split('\n')
                    
                    matching_lines = []
                    for line_num, line in enumerate(lines, 1):
                        if text_lower in line.lower():
                            matching_lines.append(line_num)
                    
                    if matching_lines:
                        results[str(file_path.relative_to(current_dir))] = matching_lines
                
                except (UnicodeDecodeError, PermissionError, IOError):
                    continue
        
        return results
    
    async def find_todos(self, path: str) -> list:
        """Find TODO, FIXME, HACK, NOTE, XXX comments in code.
        
        Args:
            path: Directory or file path to search in
        
        Returns:
            List of dicts with 'file', 'line', 'type', 'content' for each TODO
        """
        path_obj = Path(path)
        todos = []
        
        # Common TODO markers
        todo_patterns = {
            'TODO': re.compile(r'(?:#|//|/\*)\s*TODO[:\s]*(.+)', re.IGNORECASE),
            'FIXME': re.compile(r'(?:#|//|/\*)\s*FIXME[:\s]*(.+)', re.IGNORECASE),
            'HACK': re.compile(r'(?:#|//|/\*)\s*HACK[:\s]*(.+)', re.IGNORECASE),
            'NOTE': re.compile(r'(?:#|//|/\*)\s*NOTE[:\s]*(.+)', re.IGNORECASE),
            'XXX': re.compile(r'(?:#|//|/\*)\s*XXX[:\s]*(.+)', re.IGNORECASE),
        }
        
        # Determine files to search
        files_to_search = []
        if path_obj.is_file():
            files_to_search = [path_obj]
        elif path_obj.is_dir():
            # Search in common code file extensions
            code_extensions = ['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'rb', 'php']
            for ext in code_extensions:
                files_to_search.extend(path_obj.rglob(f"*.{ext}"))
            files_to_search = [f for f in files_to_search if f.is_file()]
        else:
            return [f"Error: Path is not a file or directory: {path}"]
        
        # Search for TODOs in each file
        for file_path in files_to_search:
            try:
                if self._is_binary_file(file_path):
                    continue
                
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for todo_type, pattern in todo_patterns.items():
                        match = pattern.search(line)
                        if match:
                            todos.append({
                                "file": str(file_path.relative_to(path_obj.parent) if path_obj.is_dir() else file_path),
                                "line": line_num,
                                "type": todo_type,
                                "content": match.group(1).strip()[:200] if match.group(1) else ""
                            })
                            break  # Only count once per line
            
            except (UnicodeDecodeError, PermissionError, IOError):
                continue
        
        return todos
    
    async def count_lines(self, path: str, by_extension: bool = True) -> dict:
        """Count lines of code in files.
        
        Args:
            path: Directory or file path to analyze
            by_extension: If True, group counts by file extension
        
        Returns:
            Dict with line counts, optionally grouped by extension
        """
        path_obj = Path(path)
        results = {}
        
        if path_obj.is_file():
            # Count lines in single file
            try:
                if self._is_binary_file(path_obj):
                    return {"error": "Cannot count lines in binary file"}
                
                content = path_obj.read_text(encoding="utf-8", errors="ignore")
                total_lines = len(content.splitlines())
                code_lines = len([line for line in content.splitlines() if line.strip()])
                blank_lines = total_lines - code_lines
                
                results = {
                    "file": str(path_obj),
                    "total_lines": total_lines,
                    "code_lines": code_lines,
                    "blank_lines": blank_lines
                }
            except (UnicodeDecodeError, PermissionError, IOError) as e:
                return {"error": f"Cannot read file: {str(e)}"}
        
        elif path_obj.is_dir():
            if by_extension:
                # Group by extension
                ext_counts = defaultdict(lambda: {"total": 0, "code": 0, "blank": 0, "files": 0})
                
                for file_path in path_obj.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        if self._is_binary_file(file_path):
                            continue
                        
                        ext = file_path.suffix.lstrip('.') or 'no_extension'
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        lines = content.splitlines()
                        
                        total = len(lines)
                        code = len([line for line in lines if line.strip()])
                        blank = total - code
                        
                        ext_counts[ext]["total"] += total
                        ext_counts[ext]["code"] += code
                        ext_counts[ext]["blank"] += blank
                        ext_counts[ext]["files"] += 1
                    
                    except (UnicodeDecodeError, PermissionError, IOError):
                        continue
                
                # Convert to regular dict
                results = {
                    ext: {
                        "total_lines": counts["total"],
                        "code_lines": counts["code"],
                        "blank_lines": counts["blank"],
                        "file_count": counts["files"]
                    }
                    for ext, counts in ext_counts.items()
                }
            else:
                # Total count across all files
                total_lines = 0
                code_lines = 0
                blank_lines = 0
                file_count = 0
                
                for file_path in path_obj.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    try:
                        if self._is_binary_file(file_path):
                            continue
                        
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        lines = content.splitlines()
                        
                        total = len(lines)
                        code = len([line for line in lines if line.strip()])
                        blank = total - code
                        
                        total_lines += total
                        code_lines += code
                        blank_lines += blank
                        file_count += 1
                    
                    except (UnicodeDecodeError, PermissionError, IOError):
                        continue
                
                results = {
                    "total_lines": total_lines,
                    "code_lines": code_lines,
                    "blank_lines": blank_lines,
                    "file_count": file_count
                }
        else:
            return {"error": f"Path is not a file or directory: {path}"}
        
        return results
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if a file is likely binary."""
        # Check extension
        binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.jpg', '.jpeg', 
                            '.png', '.gif', '.pdf', '.zip', '.tar', '.gz', '.pyc', '.pyo'}
        if file_path.suffix.lower() in binary_extensions:
            return True
        
        # Try to read first few bytes
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                # Check for null bytes (common in binary files)
                if b'\x00' in chunk:
                    return True
                # Check if content is mostly printable
                try:
                    chunk.decode('utf-8')
                except UnicodeDecodeError:
                    return True
        except (IOError, PermissionError):
            return True
        
        return False
