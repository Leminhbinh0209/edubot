import asyncio
from pathlib import Path
import sys
import os
from dotenv import load_dotenv
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from mybot.tools.filesystem import ReadFileTool
from mybot.tools.shell import ExecTool
from mybot.session.manager import SessionManager
from mybot.tools.registry import ToolRegistry
from mybot.agent.loop import AgentLoop
from mybot.providers.openrouter_provider import OpenRouterProvider
# Load environment variables from .env file
load_dotenv(project_root / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

async def main():
    """Main CLI entry point."""
    data_dir = Path.home() / ".mybot"
    data_dir.mkdir(exist_ok=True)
    # Initialize components
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(ExecTool())
    sessions = SessionManager(data_dir)
    agent = AgentLoop(provider, model="nvidia/nemotron-3-nano-30b-a3b:free", tools=tools, sessions=sessions)
    print("Agent ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue
        print("Agent: ", end="", flush=True)
        response = await agent.process_message(user_input)
        print(response)
        print()
if __name__ == "__main__":
    asyncio.run(main())