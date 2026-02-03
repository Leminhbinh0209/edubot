import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv(project_root / ".env")

from mybot.models import Message
from mybot.providers.openrouter_provider import OpenRouterProvider
 
def test_message_creation():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.timestamp is not None


def run_async_test(coro):
    """Helper to run async tests"""
    try:
        asyncio.run(coro)
        print("✓ Test passed")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing Message creation...")
    test_message_creation()
    print("✓ Message test passed\n")
    
