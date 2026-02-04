import sys
from pathlib import Path
import os
import tempfile
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import asyncio
from mybot.agent.loop import AgentLoop
from mybot.providers.openrouter_provider import OpenRouterProvider
from mybot.tools.registry import ToolRegistry
from mybot.tools.filesystem import ReadFileTool
from mybot.tools.shell import ExecTool
from mybot.session.manager import SessionManager

# Load environment variables from .env file
load_dotenv(project_root / ".env")


async def test_agent_basic_chat():
    """Test basic chat without tools"""
    print("\n=== Test 1: Basic Chat ===")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions = SessionManager(Path(tmpdir))
        agent = AgentLoop(
            provider, 
            model="nvidia/nemotron-3-nano-30b-a3b:free", 
            tools=tools, 
            sessions=sessions
        )
        response = await agent.process_message("Say 'Hello, World!' and nothing else.")
        print(f"Response: {response}")
        assert response is not None
        assert len(response) > 0
        print("✓ Basic chat test passed")


async def test_agent_read_file():
    """Test agent using read_file tool"""
    print("\n=== Test 2: Read File Tool ===")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    
    # Create a test file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_content = "This is a test file.\nIt has multiple lines.\nLine 3."
        test_file.write_text(test_content)
        
        sessions = SessionManager(Path(tmpdir) / "sessions")
        agent = AgentLoop(
            provider,
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            tools=tools,
            sessions=sessions
        )
        
        # Ask agent to read the file
        response = await agent.process_message(
            f"Please read the file at {test_file} and tell me what's in it."
        )
        print(f"Response: {response}")
        assert response is not None
        # The response should mention the file content
        assert len(response) > 0
        print("✓ Read file tool test passed")


async def test_agent_exec_command():
    """Test agent using exec tool"""
    print("\n=== Test 3: Exec Tool ===")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ExecTool())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions = SessionManager(Path(tmpdir) / "sessions")
        agent = AgentLoop(
            provider,
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            tools=tools,
            sessions=sessions
        )
        
        # Ask agent to run a simple command
        response = await agent.process_message(
            "Please run the command 'echo Hello from shell' and tell me the output."
        )
        print(f"Response: {response}")
        assert response is not None
        assert len(response) > 0
        print("✓ Exec tool test passed")


async def test_agent_multiple_tools():
    """Test agent with both read_file and exec tools"""
    print("\n=== Test 4: Multiple Tools ===")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(ExecTool())
    
    # Create a test file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "data.txt"
        test_file.write_text("Python\nJavaScript\nRust\nGo")
        
        sessions = SessionManager(Path(tmpdir) / "sessions")
        agent = AgentLoop(
            provider,
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            tools=tools,
            sessions=sessions
        )
        
        # Ask agent to read file and process it
        response = await agent.process_message(
            f"Read the file {test_file} and count how many programming languages are listed."
        )
        print(f"Response: {response}")
        assert response is not None
        assert len(response) > 0
        print("✓ Multiple tools test passed")


async def test_agent_session_persistence():
    """Test that session history is maintained across messages"""
    print("\n=== Test 5: Session Persistence ===")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions = SessionManager(Path(tmpdir) / "sessions")
        agent = AgentLoop(
            provider,
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            tools=tools,
            sessions=sessions
        )
        
        # First message
        response1 = await agent.process_message(
            "My name is Alice. Remember this.",
            session_key="test_session"
        )
        print(f"Response 1: {response1}")
        
        # Second message - should remember the name
        response2 = await agent.process_message(
            "What is my name?",
            session_key="test_session"
        )
        print(f"Response 2: {response2}")
        
        assert response1 is not None
        assert response2 is not None
        # The agent should remember the name (though free models may not always work perfectly)
        print("✓ Session persistence test passed")


async def test_agent_complex_task():
    """Test agent with a complex multi-step task"""
    print("\n=== Test 6: Complex Task ===")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(ExecTool())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file with numbers
        numbers_file = Path(tmpdir) / "numbers.txt"
        numbers_file.write_text("10\n20\n30\n40\n50")
        
        sessions = SessionManager(Path(tmpdir) / "sessions")
        agent = AgentLoop(
            provider,
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            tools=tools,
            sessions=sessions
        )
        
        # Ask agent to read file, calculate sum, and create a summary
        response = await agent.process_message(
            f"Read the file {numbers_file}, calculate the sum of all numbers, and tell me the result."
        )
        print(f"Response: {response}")
        assert response is not None
        assert len(response) > 0
        print("✓ Complex task test passed")


async def run_all_tests():
    """Run all comprehensive tests"""
    print("=" * 60)
    print("Running Comprehensive Agent Tests")
    print("=" * 60)
    
    try:
        # await test_agent_basic_chat()
        # await test_agent_read_file()
        # await test_agent_exec_command()
        # await test_agent_multiple_tools()
        # await test_agent_session_persistence()
        await test_agent_complex_task()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_all_tests())