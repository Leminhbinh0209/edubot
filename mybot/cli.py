import asyncio
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from mybot.tools.filesystem import ReadFileTool, WriteFileTool
from mybot.tools.shell import ExecTool
from mybot.tools.code_analysis import CodeAnalysisTool
from mybot.tools.search import SearchTool
from mybot.tools.test_runner import TestTool
from mybot.session.manager import SessionManager
from mybot.tools.registry import ToolRegistry
from mybot.agent.loop import AgentLoop
from mybot.providers.openrouter_provider import OpenRouterProvider
from mybot.utils.stream_smoother import StreamSmoother

# Load environment variables from .env file
load_dotenv(project_root / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"


async def main():
    """Main CLI entry point."""
    data_dir = Path.home() / ".mybot"
    data_dir.mkdir(exist_ok=True)
    # Initialize components
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()

    # Register basic tools
    tools.register(ReadFileTool())
    tools.register(WriteFileTool())
    tools.register(ExecTool())

    # Register advanced tools
    tools.register(CodeAnalysisTool())
    tools.register(SearchTool())
    tools.register(TestTool())

    sessions = SessionManager(data_dir)
    # Enable planning mode for better multi-step task handling
    agent = AgentLoop(provider, model=MODEL, tools=tools, sessions=sessions, use_planning=True)
    print("Agent ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue
        print("Agent: ", end="", flush=True)

        # Define callback to print chunks as they arrive
        def print_chunk(chunk: str):
            print(chunk, end="", flush=True)

        # Thoughtful/deliberate (slower)
        smoother = StreamSmoother(
            callback=print_chunk,
            min_chunk_chars=4,
            max_chunk_chars=100,
            base_delay=0.04,
            char_delay=0.01,
        )

        # Create async callback wrapper for StreamSmoother
        async def smooth_callback(chunk: str):
            await smoother.push(chunk)

        response = await agent.process_message(
            user_input, stream=True, stream_callback=smooth_callback
        )

        # Flush any remaining buffered content
        await smoother.flush_final()
        print()  # New line after response
        print()


if __name__ == "__main__":
    asyncio.run(main())
