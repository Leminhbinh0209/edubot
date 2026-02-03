from pathlib import Path
from mybot.tools.base import Tool

class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                }
            },
            "required": ["path"]
        }
    
    async def execute(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {str(e)}"