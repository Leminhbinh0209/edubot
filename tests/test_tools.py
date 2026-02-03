# tests/test_tools.py
from pathlib import Path
import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mybot.tools.registry import ToolRegistry
from mybot.tools.filesystem import ReadFileTool
import asyncio

async def test_tool_registry():
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    
    assert registry.get("read_file") is not None
    assert len(registry.get_definitions()) == 1


async def test_read_file():
    tool = ReadFileTool()
    # Create test file
    test_file = Path("/tmp/test.txt")
    test_file.write_text("Hello, World!\nThis test file is for testing the read file tool.")

    result = await tool.execute(path=str(test_file))
    print(result)
    assert "Hello, World!" in result
    print("✓ Test passed")
if __name__ == "__main__":
    print("Testing tool registry...")
    asyncio.run(test_tool_registry())
    print("Testing read file...")
    asyncio.run(test_read_file())
    print("✓ All tests passed")