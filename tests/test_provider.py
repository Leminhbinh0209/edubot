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
 

async def test_openrouter_basic_chat():
    """Test basic chat functionality with OpenRouter API"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    
    # Test with a simple message
    messages = [
        {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
    ]
    
    # Use a free model for testing
    response = await provider.chat(
        messages=messages,
        model="nvidia/nemotron-3-nano-30b-a3b:free"
    )
    
    assert response is not None
    assert response.content is not None
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.finish_reason == "stop"
    assert not response.has_tool_calls()
    
    print(f"Response: {response.content}")
    
    # Clean up
    await provider.client.aclose()

async def test_openrouter_with_tools():
    """Test OpenRouter API with tool calls"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")
    
    provider = OpenRouterProvider(api_key=api_key)
    
    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ]
    
    # Use a model that supports tool calling
    # Note: Most models that support tool calling require API credits
    # Common options: nvidia/nemotron-3-nano-30b-a3b:free, openai/gpt-4, anthropic/claude-3-haiku
    # If you don't have credits, you can skip this test or use a model that supports tools
    response = await provider.chat(
        messages=messages,
        tools=tools,
        model="nvidia/nemotron-3-nano-30b-a3b:free"  # This model supports tool calling (requires credits)
    )
    
    assert response is not None
    # The response might have tool calls or content
    if response.has_tool_calls:
        assert len(response.tool_calls) > 0
        tool_call = response.tool_calls[0]
        assert tool_call.name == "get_weather"
        assert "location" in tool_call.args
        print(f"Tool call: {tool_call.name} with args: {tool_call.args}")
    else:
        assert response.content is not None
        print(f"Response: {response.content}")
    
    # Clean up
    await provider.client.aclose()

def run_async_test(coro):
    """Helper to run async tests"""
    try:
        asyncio.run(coro)
        print("✓ Test passed")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            print(f"✗ Test failed: Model not found (404). The model may not exist or be unavailable.")
            print(f"  Error details: {e}")
        elif "401" in error_msg or "403" in error_msg:
            print(f"✗ Test failed: Authentication error. Check your API key.")
            print(f"  Error details: {e}")
        elif "402" in error_msg or "insufficient" in error_msg.lower():
            print(f"✗ Test failed: Insufficient credits. This model requires API credits.")
            print(f"  Error details: {e}")
        else:
            print(f"✗ Test failed: {e}")
        raise

if __name__ == "__main__":

    print("Testing OpenRouter basic chat...")
    run_async_test(test_openrouter_basic_chat())
    print()
    
    print("Testing OpenRouter with tools...")
    run_async_test(test_openrouter_with_tools())
    print("\nAll tests passed!")