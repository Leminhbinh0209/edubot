# Building AI Assistant from Scratch: Implementation Guide

Build a lightweight AI assistant with tool calling, session management, and CLI interface in 5 phases.

## 🎯 Overview

**Goal**: Build an AI assistant that can:
- Chat with users via CLI
- Execute tools (read files, run commands)
- Maintain conversation history
- Use LLM providers for reasoning

**Timeline**: 1 hours for basic version.

## 📋 Prerequisites

- Python 3.11+
- Basic async/await knowledge
- Understanding of LLM APIs
- Familiarity with type hints

## 🏗️ Architecture

```
edubot/
├── mybot/
│   ├── agent/      # Orchestrates everything
│   ├── providers/  # LLM API connections
│   ├── tools/      # Extensible tool system
│   └── session/    # Conversation memory
├── tests/
└── README.md
```

---
# PART I: BASIC COMPONENTS
## Phase 1: Core Data Models

**Why First**: Everything else depends on these foundational structures.

<details>
<summary><b>Click to expand: Implementation</b></summary>

**Files**: `mybot/models.py`, `mybot/exceptions.py`

```python
# mybot/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class Message:
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]

@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = None
    finish_reason: str = "stop"
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
    
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
```

```python
# mybot/exceptions.py
class BotException(Exception):
    pass

class LLMError(BotException):
    pass

class ToolError(BotException):
    pass
```

</details>

<details>
<summary><b>Click to expand: Test</b></summary>

```python
# tests/test_models.py
def test_message_creation():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.timestamp is not None
```

</details>

---

## Phase 2: LLM Provider

**Why Second**: Agent needs LLM intelligence to function.

**Important**: Use **dict-based messages**, not Message objects: `list[dict[str, Any]]` where each dict has `{"role": "user", "content": "..."}`.

<details>
<summary><b>Click to expand: Implementation</b></summary>

**Files**: `mybot/providers/base.py`, `mybot/providers/openrouter_provider.py`

```python
# mybot/providers/base.py
from abc import ABC, abstractmethod
from typing import Any
from mybot.models import LLMResponse

class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        pass
```

```python
# mybot/providers/openrouter_provider.py
import httpx
import json
from typing import Any
from mybot.providers.base import LLMProvider
from mybot.models import LLMResponse, ToolCall

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    async def chat(self, messages, tools=None, model=None) -> LLMResponse:
        payload = {
            "model": model or "nvidia/nemotron-3-nano-30b-a3b:free",
            "messages": messages,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        
        choice = data["choices"][0]
        message = choice["message"]
        
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    args=args
                ))
        
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop")
        )
```

</details>

<details>
<summary><b>Click to expand: Setup & Test</b></summary>

1. **Set up API Key**:
   ```bash
   # Create .env file
   echo "OPENROUTER_API_KEY=sk-or-v1-your-api-key-here" > .env
   ```
   Get your API key from [openrouter.ai/keys](https://openrouter.ai/keys)

2. **Install Dependencies**:
   ```bash
   pip install python-dotenv httpx
   ```

3. **Run Test**:
   ```bash
   python tests/test_provider.py
   ```

**Expected Output**:
```
Testing OpenRouter basic chat...
Response: Hello, World!
✓ Test passed

Testing OpenRouter with tools...
Tool call: get_weather with args: {'location': 'San Francisco, CA'}
✓ Test passed
```

**Troubleshooting**:
- **404 Error**: Model not found - check model name
- **401/403 Error**: Authentication failed - verify API key
- **402 Error**: Insufficient credits - some models require credits

</details>

---

## Phase 3: Tool System

**Why Third**: Agent needs tools to interact with the environment.

<details>
<summary><b>Click to expand: Implementation</b></summary>

**Files**: `mybot/tools/base.py`, `mybot/tools/registry.py`, `mybot/tools/filesystem.py`, `mybot/tools/shell.py`

```python
# mybot/tools/base.py
from abc import ABC, abstractmethod
from typing import Any

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        pass
    
    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

```python
# mybot/tools/registry.py
from typing import Any
from mybot.tools.base import Tool

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        try:
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
```

```python
# mybot/tools/filesystem.py
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
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }
    
    async def execute(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File not found: {path}"
        return file_path.read_text(encoding="utf-8")
```

```python
# mybot/tools/shell.py
import asyncio
import os
from typing import Any
from mybot.tools.base import Tool

class ExecTool(Tool):
    def __init__(self, timeout: int = 60, working_dir: str | None = None):
        self.timeout = timeout
        self.working_dir = working_dir
    
    @property
    def name(self) -> str:
        return "exec"
    
    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "working_dir": {"type": "string", "description": "Optional working directory"}
            },
            "required": ["command"]
        }
    
    async def execute(self, command: str, working_dir: str | None = None, **kwargs: Any) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            output_parts = []
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")
            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")
            
            result = "\n".join(output_parts) if output_parts else "(no output)"
            return result[:10000] + ("\n... (truncated)" if len(result) > 10000 else "")
        except asyncio.TimeoutError:
            return f"Error: Command timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
```

</details>

---

## Phase 4: Session Management

**Why Fourth**: Agent needs context from previous messages.

<details>
<summary><b>Click to expand: Implementation</b></summary>

**File**: `mybot/session/manager.py`

```python
import json
from pathlib import Path
from datetime import datetime
from typing import Any

class Session:
    def __init__(self, key: str):
        self.key = key
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": msg["role"], "content": msg["content"]} for msg in recent]

class SessionManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sessions_dir = data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Session] = {}
    
    def get_session_path(self, key: str) -> Path:
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.sessions_dir / f"{safe_key}.json"
    
    def get_or_create(self, key: str) -> Session:
        if key in self._cache:
            return self._cache[key]
        path = self.get_session_path(key)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                session = Session(key=key)
                session.messages = data.get("messages", [])
                if data.get("created_at"):
                    session.created_at = datetime.fromisoformat(data["created_at"])
                self._cache[key] = session
                return session
            except Exception:
                pass
        session = Session(key=key)
        self._cache[key] = session
        return session
    
    def save(self, session: Session) -> None:
        path = self.get_session_path(session.key)
        data = {
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": session.messages
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._cache[session.key] = session
```

</details>

---

## Phase 5: Agent Loop

**Why Fifth**: Core functionality - brings everything together.

<details>
<summary><b>Click to expand: Implementation</b></summary>

**File**: `mybot/agent/loop.py`

```python
import asyncio
from pathlib import Path
from mybot.providers.base import LLMProvider
from mybot.tools.registry import ToolRegistry
from mybot.session.manager import SessionManager

class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        model: str = "nvidia/nemotron-3-nano-30b-a3b:free",
        tools: ToolRegistry = None,
        sessions: SessionManager = None,
        max_iterations: int = 10,
    ):
        if tools is None:
            tools = ToolRegistry()
        if sessions is None:
            sessions = SessionManager(Path.home() / ".mybot" / "sessions")
        self.provider = provider
        self.tools = tools
        self.sessions = sessions
        self.max_iterations = max_iterations
        self.model = model
    
    async def process_message(self, user_message: str, session_key: str = "default") -> str:
        import json
        session = self.sessions.get_or_create(session_key)
        messages = []
        system_prompt = """You are a helpful AI assistant. You have access to tools.
        When you need to use a tool, call it. Otherwise, respond directly to the user."""
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(session.get_history())
        messages.append({"role": "user", "content": user_message})
        
        iteration = 0
        final_response = None
        
        while iteration < self.max_iterations:
            iteration += 1
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls():
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": tool_call_dicts
                })
                
                for tool_call in response.tool_calls:
                    result = await self.tools.execute(tool_call.name, tool_call.args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result
                    })
            else:
                final_response = response.content
                break
        
        if final_response is None:
            final_response = "I'm having trouble processing that request."
        
        session.add_message("user", user_message)
        session.add_message("assistant", final_response)
        self.sessions.save(session)
        return final_response
```

**Tip**: Uncomment print statements in the loop to see step-by-step execution.

</details>

<details>
<summary><b>Click to expand: Test</b></summary>

```python
# tests/test_agent.py
async def test_agent_complex_task():
    api_key = os.getenv("OPENROUTER_API_KEY")
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(ExecTool())
    
    with tempfile.TemporaryDirectory() as tmpdir:
        numbers_file = Path(tmpdir) / "numbers.txt"
        numbers_file.write_text("10\n20\n30\n40\n50")
        
        sessions = SessionManager(Path(tmpdir) / "sessions")
        agent = AgentLoop(provider, model="nvidia/nemotron-3-nano-30b-a3b:free", tools=tools, sessions=sessions)
        
        response = await agent.process_message(
            f"Read the file {numbers_file}, calculate the sum of all numbers, and tell me the result."
        )
        assert response is not None
```

</details>

---

## Phase 6: CLI Interface

**Why Sixth**: Need a way to interact with the agent.

<details>
<summary><b>Click to expand: Implementation</b></summary>

**File**: `mybot/cli.py`

```python
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import os
from mybot.providers.openrouter_provider import OpenRouterProvider
from mybot.tools.registry import ToolRegistry
from mybot.tools.filesystem import ReadFileTool
from mybot.tools.shell import ExecTool
from mybot.session.manager import SessionManager
from mybot.agent.loop import AgentLoop

load_dotenv()

async def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return
    
    data_dir = Path.home() / ".mybot"
    data_dir.mkdir(exist_ok=True)
    
    provider = OpenRouterProvider(api_key=api_key)
    tools = ToolRegistry()
    tools.register(ReadFileTool())
    tools.register(ExecTool())
    sessions = SessionManager(data_dir)
    agent = AgentLoop(
        provider,
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        tools=tools,
        sessions=sessions
    )
    
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
```

</details>

---

## 🚀 Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <your-repo>
   cd edubot
   pip install python-dotenv httpx
   ```

2. **Configure API Key**:
   ```bash
   echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env
   ```

3. **Run Tests**:
   ```bash
   python tests/test_provider.py
   python tests/test_agent.py
   ```

4. **Start CLI**:
   ```bash
   python mybot/cli.py
   ```

## 📚 Key Concepts

- **Modular Design**: Each component (provider, tools, sessions) is independent
- **Tool Calling**: LLM decides when to use tools based on user requests
- **Session Persistence**: Conversations saved to disk for context across restarts
- **Extensible**: Add new tools by implementing the `Tool` interface

## 🎯 Next Steps
- Add streaming responses
- Add more tools (web search, database queries, APIs)
- Improve error handling and retry logic
- Build a web interface
- Add multi-modal capabilities

---
# PART II: IMPROVEMENT
## Streaming Response

**Why**: Improve user experience by displaying responses in real-time instead of waiting for the complete response.

This implementation adds streaming support to display text as it's generated, making the interface feel more responsive and interactive.

<details>
<summary><b>Click to expand: Overview</b></summary>

**What Changed**:
1. **Base Provider Interface**: Added `chat_stream()` method for streaming support
2. **OpenRouterProvider**: Implemented Server-Sent Events (SSE) streaming
3. **AgentLoop**: Added streaming support with async callbacks
4. **StreamSmoother**: Utility class for smooth, natural-feeling output
5. **CLI**: Integrated streaming with smooth output display

**Key Features**:
- Real-time text display as it's generated
- Smooth output with intelligent chunking at natural boundaries
- Automatic fallback to non-streaming for tool calls
- Configurable smoothing parameters

</details>

<details>
<summary><b>Click to expand: Base Provider Interface</b></summary>

**File**: `mybot/providers/base.py`

Added `chat_stream()` method to the base `LLMProvider` class:

```python
async def chat_stream(
    self,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream chat completion response.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        tools: Optional list of tool definitions in OpenAI format.
        model: Model identifier.
        
    Yields:
        Text chunks as they are generated.
        
    Note:
        This is an optional method. Providers that don't support streaming
        can fall back to the regular chat() method.
    """
    # Default implementation: fall back to non-streaming
    response = await self.chat(messages, tools, model)
    if response.content:
        yield response.content
```

**Key Points**:
- Returns an async iterator that yields text chunks
- Default implementation falls back to non-streaming for compatibility
- Providers can override this method to implement actual streaming

</details>

<details>
<summary><b>Click to expand: OpenRouterProvider Streaming</b></summary>

**File**: `mybot/providers/openrouter_provider.py`

Implemented `chat_stream()` method following OpenRouter's Server-Sent Events (SSE) format:

```python
async def chat_stream(
    self,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream chat completion response from OpenRouter API.
    
    Note: Streaming is only supported for non-tool responses.
    If tools are provided, this will fall back to non-streaming.
    """
    # Fall back to non-streaming for tool calls
    if tools:
        response = await self.chat(messages, tools, model)
        if response.content:
            yield response.content
        return
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "stream": True,  # Enable streaming
    }
    
    buffer = ""
    async with self.client.stream("POST", "/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for chunk in response.aiter_text():
            buffer += chunk
            while True:
                try:
                    # Find the next complete SSE line
                    line_end = buffer.find('\n')
                    if line_end == -1:
                        break
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            return
                        try:
                            data_obj = json.loads(data)
                            content = data_obj["choices"][0]["delta"].get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    break
```

**Key Implementation Details**:
- Uses `httpx.AsyncClient.stream()` for streaming HTTP requests
- Parses Server-Sent Events (SSE) format with `data: ` prefix
- Handles incomplete lines with a buffer mechanism
- Extracts content from `delta.content` in the response
- Falls back to non-streaming when tools are provided (tool calls require full response)

</details>

<details>
<summary><b>Click to expand: AgentLoop Streaming Support</b></summary>

**File**: `mybot/agent/loop.py`

Updated `process_message()` to support streaming:

```python
async def process_message(
    self,
    user_message: str,
    session_key: str = "default",
    stream: bool = False,
    stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None,
) -> str:
    """Process a user message and return response."""
    # ... existing code ...
    
    # For final responses, use streaming if enabled
    if stream and stream_callback:
        final_response = ""
        async for chunk in self.provider.chat_stream(
            messages=messages,
            tools=None,  # No tools for final response
            model=self.model
        ):
            final_response += chunk
            # Support both sync and async callbacks
            if asyncio.iscoroutinefunction(stream_callback):
                await stream_callback(chunk)
            else:
                stream_callback(chunk)
    else:
        final_response = response.content
```

**Key Features**:
- `stream` parameter to enable/disable streaming
- `stream_callback` parameter for handling chunks (supports both sync and async)
- Streaming only used for final responses (no tool calls)
- Tool call iterations still use non-streaming (required for parsing tool_calls)

</details>

<details>
<summary><b>Click to expand: StreamSmoother Utility</b></summary>

**File**: `mybot/utils/stream_smoother.py`

A utility class that makes streaming output feel more natural by:
- Buffering chunks until natural boundaries (punctuation, newlines)
- Adding configurable delays between chunks
- Respecting min/max chunk sizes

```python
class StreamSmoother:
    def __init__(
        self,
        callback: Callable[[str], None],
        min_chunk_chars: int = 15,
        max_chunk_chars: int = 80,
        base_delay: float = 0.03,
        char_delay: float = 0.008,  # delay per character
    ):
        self.callback = callback
        self.buffer = ""
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.base_delay = base_delay
        self.char_delay = char_delay
        self.boundary_re = re.compile(r'[.!?]\s+|[,;:]\s+|\n+')
        self.first_flush = True
    
    async def push(self, text: str):
        """Add text to buffer and flush when appropriate."""
        self.buffer += text
        
        # Always flush on max chars
        if len(self.buffer) >= self.max_chunk_chars:
            await self._flush_at_boundary()
            return
        
        # Flush at natural boundaries if we have enough content
        if len(self.buffer) >= self.min_chunk_chars:
            match = self.boundary_re.search(self.buffer)
            if match:
                await self._flush_at_boundary(match.end())
    
    async def _flush_at_boundary(self, pos: int = None):
        """Flush buffered content at natural boundaries."""
        if not self.buffer:
            return
        
        # Determine what to flush
        if pos is None:
            to_flush = self.buffer
            self.buffer = ""
        else:
            to_flush = self.buffer[:pos]
            self.buffer = self.buffer[pos:]
        
        # First chunk is instant, then add natural delays
        if not self.first_flush:
            # Delay based on chunk length for natural feel
            delay = self.base_delay + (len(to_flush) * self.char_delay)
            await asyncio.sleep(min(delay, 0.15))  # cap at 150ms
        
        self.callback(to_flush)
        self.first_flush = False
    
    async def flush_final(self):
        """Flush any remaining buffer"""
        if self.buffer:
            if not self.first_flush:
                await asyncio.sleep(self.base_delay)
            self.callback(self.buffer)
            self.buffer = ""
```

**Configuration Parameters**:
- `min_chunk_chars`: Minimum characters before flushing at boundaries (default: 15)
- `max_chunk_chars`: Maximum characters before forcing a flush (default: 80)
- `base_delay`: Base delay between chunks in seconds (default: 0.03)
- `char_delay`: Additional delay per character (default: 0.008)

**How It Works**:
1. Buffers incoming text chunks
2. Flushes when:
   - Buffer reaches `max_chunk_chars` (forced flush)
   - Natural boundary found (punctuation, newlines) AND buffer >= `min_chunk_chars`
3. Adds delays between chunks for natural feel
4. First chunk appears instantly for responsiveness

</details>

<details>
<summary><b>Click to expand: CLI Integration</b></summary>

**File**: `mybot/cli.py`

Updated CLI to use streaming with StreamSmoother:

```python
from mybot.utils.stream_smoother import StreamSmoother

async def main():
    # ... initialization code ...
    
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
        
        # Create StreamSmoother for smoother output
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
            user_input,
            stream=True,
            stream_callback=smooth_callback
        )
        
        # Flush any remaining buffered content
        await smoother.flush_final()
        print()  # New line after response
        print()
```

**Usage**:
- Streaming is enabled by default in the CLI
- Text appears in real-time as it's generated
- Output is smoothed for natural reading experience
- Tool calls still display normally (non-streaming)

</details>

<details>
<summary><b>Click to expand: Example Usage</b></summary>

**Before (Non-Streaming)**:
```
You: Write a short story about a robot
Agent: [waits for complete response...]
Agent: Once upon a time, there was a robot...
```

**After (Streaming)**:
```
You: Write a short story about a robot
Agent: Once upon a time, there was a robot... [text appears progressively]
```

**With Tool Calls**:
```
You: Read the file data.txt and summarize it
Agent:   [1] Tool: read_file
         Args: {'path': 'data.txt'}
  [1] Executing: read_file({'path': 'data.txt'})
      Result: [file contents...]
Agent: The file contains... [streaming response]
```

</details>

<details>
<summary><b>Click to expand: Customization</b></summary>

**Adjust StreamSmoother Parameters**:

For faster, more responsive output:
```python
smoother = StreamSmoother(
    callback=print_chunk,
    min_chunk_chars=4,
    max_chunk_chars=100,
    base_delay=0.02,  # Faster
    char_delay=0.005,  # Less delay per char
)
```

For slower, more deliberate output:
```python
smoother = StreamSmoother(
    callback=print_chunk,
    min_chunk_chars=15,
    max_chunk_chars=80,
    base_delay=0.05,  # Slower
    char_delay=0.012,  # More delay per char
)
```

**Disable Streaming**:
```python
response = await agent.process_message(
    user_input,
    stream=False  # Disable streaming
)
```

</details>

## Advanced Tools

This section documents three powerful tools that extend the basic functionality of the AI assistant: code analysis, advanced search, and test running capabilities.

<details>
<summary><b>Click to expand: Code Analysis Tool</b></summary>

**File**: `mybot/tools/code_analysis.py`

The `CodeAnalysisTool` provides code quality analysis capabilities including linting, formatting, and type checking.

**Supported Actions**:
- `pylint`: Run pylint linter and get analysis results
- `black_format`: Auto-format Python code using black
- `mypy_check`: Run mypy type checker on Python code

**Usage Example**:
```python
from mybot.tools.code_analysis import CodeAnalysisTool

tool = CodeAnalysisTool()

# Run pylint
result = await tool.execute(action="pylint", file_path="myfile.py")
# Returns: JSON string with score, errors, warnings, messages

# Format code with black
result = await tool.execute(action="black_format", file_path="myfile.py")
# Returns: Formatted code or "already formatted" message

# Check types with mypy
result = await tool.execute(action="mypy_check", file_path="myfile.py")
# Returns: JSON string with errors, warnings, notes, exit_code
```

**Pylint Output Format**:
```json
{
  "score": 8.5,
  "errors": 2,
  "warnings": 5,
  "messages": [...],
  "raw_output": "..."
}
```

**Mypy Output Format**:
```json
{
  "errors": [
    {"file": "myfile.py", "line": 10, "message": "...", "code": "..."}
  ],
  "warnings": [...],
  "notes": [...],
  "exit_code": 1
}
```

**Key Features**:
- Graceful handling when tools are not installed
- Configurable timeout (default: 60 seconds)
- Structured JSON output for easy parsing
- Automatic error detection and reporting

**Installation Requirements**:
- `pylint`: `pip install pylint`
- `black`: `pip install black`
- `mypy`: `pip install mypy`

</details>

<details>
<summary><b>Click to expand: Search Tool</b></summary>

**File**: `mybot/tools/search.py`

The `SearchTool` provides advanced file search and code analysis capabilities, going beyond basic file reading.

**Supported Actions**:
- `grep`: Search for patterns in files using regex
- `find_files`: Find files matching a pattern/name
- `find_in_files`: Search for text in files with specific extensions
- `find_todos`: Find TODO, FIXME, HACK, NOTE, XXX comments in code
- `count_lines`: Count lines of code, optionally grouped by file extension

**Usage Example**:
```python
from mybot.tools.search import SearchTool

tool = SearchTool()

# Grep for pattern
result = await tool.execute(action="grep", path="./src", pattern="def.*test", recursive=True)
# Returns: List of matches with file, line, content

# Find files
result = await tool.execute(action="find_files", path="./src", pattern="*.py")
# Returns: List of matching file paths

# Find text in specific file types
result = await tool.execute(action="find_in_files", text="TODO", extensions=["py", "js"])
# Returns: Dict mapping file paths to line numbers

# Find TODO comments
result = await tool.execute(action="find_todos", path="./src")
# Returns: List of TODO markers with file, line, type, content

# Count lines of code
result = await tool.execute(action="count_lines", path="./src")
# Returns: Dict with line counts grouped by extension
```

**Grep Output Format**:
```json
[
  {
    "file": "src/main.py",
    "line": 42,
    "content": "def test_function():"
  },
  ...
]
```

**Find TODOs Output Format**:
```json
[
  {
    "file": "src/main.py",
    "line": 10,
    "type": "TODO",
    "content": "Fix this function"
  },
  {
    "file": "src/main.py",
    "line": 15,
    "type": "FIXME",
    "content": "Add error handling"
  }
]
```

**Count Lines Output Format**:
```json
{
  "py": {
    "total_lines": 1500,
    "code_lines": 1200,
    "blank_lines": 300,
    "file_count": 25
  },
  "js": {
    "total_lines": 800,
    "code_lines": 650,
    "blank_lines": 150,
    "file_count": 12
  }
}
```

**Key Features**:
- Regex pattern support for grep
- Recursive directory searching
- Binary file detection and skipping
- Natural boundary detection for TODOs
- Extension-based line counting
- Handles large codebases efficiently

</details>

<details>
<summary><b>Click to expand: Test Runner Tool</b></summary>

**File**: `mybot/tools/test_runner.py`

The `TestTool` provides comprehensive test running capabilities for multiple testing frameworks and coverage reporting.

**Supported Actions**:
- `run_pytest`: Run pytest tests (Python)
- `run_jest`: Run jest tests (JavaScript/TypeScript)
- `run_unittest`: Run Python unittest module
- `coverage_report`: Generate code coverage report

**Usage Example**:
```python
from mybot.tools.test_runner import TestTool

tool = TestTool()

# Run pytest
result = await tool.execute(action="run_pytest", path="./tests", verbose=True)
# Returns: JSON string with passed, failed, skipped, errors, output, exit_code

# Run jest
result = await tool.execute(action="run_jest", path="./tests")
# Returns: JSON string with passed, failed, skipped, output, exit_code

# Run unittest
result = await tool.execute(action="run_unittest", module="tests.test_example")
# Returns: JSON string with passed, failed, skipped, errors, output, exit_code

# Generate coverage report
result = await tool.execute(action="coverage_report")
# Returns: JSON string with coverage_percent, lines_covered, lines_total, branches_covered
```

**Pytest/Jest/Unittest Output Format**:
```json
{
  "passed": 45,
  "failed": 2,
  "skipped": 3,
  "errors": 0,
  "output": "...",
  "exit_code": 1
}
```

**Coverage Report Output Format**:
```json
{
  "coverage_percent": 85,
  "lines_covered": 1200,
  "lines_total": 1412,
  "branches_covered": 450,
  "output": "..."
}
```

**Key Features**:
- Supports multiple test frameworks (pytest, jest, unittest)
- Automatic output parsing to extract test counts
- Configurable timeout (default: 300 seconds for tests)
- Coverage report generation (supports pytest-cov and coverage.py)
- Verbose/quiet modes for pytest
- Graceful handling when test tools are not installed

**Installation Requirements**:
- `pytest`: `pip install pytest`
- `pytest-cov`: `pip install pytest-cov` (for coverage)
- `coverage`: `pip install coverage` (alternative coverage tool)
- `jest`: `npm install --save-dev jest` (for JavaScript/TypeScript tests)

</details>

<details>
<summary><b>Click to expand: Registering Advanced Tools</b></summary>

To use these tools in your agent, register them in your CLI or agent setup:

```python
from mybot.tools.code_analysis import CodeAnalysisTool
from mybot.tools.search import SearchTool
from mybot.tools.test_runner import TestTool
from mybot.tools.registry import ToolRegistry

# Create tool registry
tools = ToolRegistry()

# Register basic tools
tools.register(ReadFileTool())
tools.register(ExecTool())

# Register advanced tools
tools.register(CodeAnalysisTool())
tools.register(SearchTool())
tools.register(TestTool())

# Use with agent
agent = AgentLoop(provider, tools=tools, sessions=sessions)
```

The agent will automatically have access to all registered tools and can use them based on user requests.

</details>

---
## Planning

**Why**: Enable the agent to break down complex, multi-step tasks into actionable plans, execute them systematically, and synthesize results into coherent responses.

The planning system transforms the agent from reactive tool-calling to proactive task orchestration, allowing it to handle complex requests that require multiple steps, dependencies, and data flow between steps.

<details>
<summary><b>Click to expand: Overview</b></summary>

**What is Planning Mode?**

Planning mode is a three-phase approach to handling complex tasks:

1. **Plan Phase**: The LLM analyzes the user's request and creates a detailed, step-by-step execution plan
2. **Execute Phase**: The agent executes the plan, respecting dependencies and parallelizing independent steps
3. **Synthesize Phase**: The agent summarizes execution results into a natural language response

**Key Features**:
- Automatic plan generation from natural language requests
- Dependency management between steps
- Parallel execution of independent steps for performance
- Dynamic argument resolution from previous step results
- Plan validation and auto-fixing
- Retry logic for failed steps
- Comprehensive error handling

**When to Use Planning**:
- Multi-step tasks (e.g., "Find all TODOs, then run pylint on those files")
- Tasks with dependencies (e.g., "Read file X, analyze it, then write results to file Y")
- Complex workflows (e.g., "Find test files, run tests, generate coverage report")
- Tasks requiring data flow between steps

</details>

<details>
<summary><b>Click to expand: Enabling Planning Mode</b></summary>

**File**: `mybot/cli.py` or `mybot/agent/loop.py`

Planning mode is enabled by setting `use_planning=True` when creating the `AgentLoop`:

```python
from mybot.agent.loop import AgentLoop
from mybot.providers.openrouter_provider import OpenRouterProvider
from mybot.tools.registry import ToolRegistry

# Initialize agent with planning enabled
agent = AgentLoop(
    provider=provider,
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    tools=tools,
    sessions=sessions,
    use_planning=True,  # Enable planning mode
    verbose=True,        # Show detailed execution logs
    max_retries=2         # Retry failed steps up to 2 times
)
```

**Configuration Options**:
- `use_planning`: Enable/disable planning mode (default: `False`)
- `verbose`: Show detailed execution logs (default: `True`)
- `max_retries`: Number of retries for failed steps (default: `2`)

</details>

<details>
<summary><b>Click to expand: Phase 1 - Planning</b></summary>

**How It Works**:

The planning phase uses the LLM to analyze the user's request and generate a structured execution plan:

```python
async def _plan(self, user_message: str, session) -> list[dict]:
    """Create execution plan using LLM."""
    # 1. Build detailed tool schemas
    # 2. Send planning prompt to LLM
    # 3. Extract JSON plan from response
    # 4. Validate plan structure
    # 5. Auto-fix common issues
    # 6. Return validated plan
```

**Plan Structure**:

Each plan consists of steps with the following structure:

```json
{
  "steps": [
    {
      "step": 1,
      "tool": "search",
      "args": {
        "action": "find_todos",
        "path": "mybot"
      },
      "reasoning": "Find all TODO comments in the codebase",
      "depends_on": []
    },
    {
      "step": 2,
      "tool": "code_analysis",
      "args": {
        "action": "pylint",
        "file_path": "{{step1.output[0]}}"
      },
      "reasoning": "Run pylint on the first file with TODOs",
      "depends_on": [1]
    }
  ]
}
```

**Plan Validation**:

The system validates plans before execution:
- ✅ Step numbers are valid and unique
- ✅ Tools exist in the registry
- ✅ Dependencies don't reference future steps
- ✅ Required parameters are present

**Auto-Fixing**:

Common plan issues are automatically fixed:
- Renumbering steps sequentially
- Removing steps with invalid tools
- Fixing dependency references

**Example Output**:

```
📋 PHASE 1: Creating Execution Plan...
------------------------------------------------------------
   Generated 3 step(s)

✅ Plan created with 3 step(s)

📝 Execution Plan:
   Step 1: Find all TODO comments in mybot directory
   Step 2: Run pylint on files with TODOs (depends on: [1])
   Step 3: Generate coverage report (depends on: [2])
```

</details>

<details>
<summary><b>Click to expand: Phase 2 - Execution</b></summary>

**How It Works**:

The execution phase runs the plan steps, respecting dependencies and parallelizing when possible:

```python
async def _execute_plan(self, plan: list[dict]) -> dict:
    """Execute plan steps, parallelizing independent steps."""
    # 1. Find steps ready to execute (dependencies met)
    # 2. Execute ready steps in parallel
    # 3. Collect results
    # 4. Repeat until all steps complete
```

**Key Features**:

1. **Dependency Management**: Steps only execute when their dependencies are complete
2. **Parallel Execution**: Independent steps run simultaneously for better performance
3. **Dynamic Argument Resolution**: Arguments are automatically extracted from previous step results
4. **Retry Logic**: Failed steps are retried up to `max_retries` times
5. **Error Detection**: Tool-specific error detection distinguishes real errors from legitimate outputs

**Dynamic Argument Resolution**:

The system automatically extracts data from previous step results:

```python
# Step 1: find_files returns ["test1.py", "test2.py"]
# Step 2: Can use {{step1.output[0]}} to reference first file
# System automatically resolves to "test1.py"
```

**Supported Patterns**:
- Template placeholders: `{{step1.output[0]}}`
- Automatic extraction from formatted output
- Path construction from relative filenames
- JSON parsing from structured output

**Example Output**:

```
⚙️  PHASE 2: Executing Plan...
============================================================

📍 Step 1: Find all TODO comments in mybot directory
   🔧 Tool: search
   📥 Args: {
      "action": "find_todos",
      "path": "mybot"
   }
   ✅ Success
   📤 Result preview: Found 5 results:
   mybot/providers/base.py:10: TODO: Add caching
   ...

📍 Step 2: Run pylint on files with TODOs
   🔧 Tool: code_analysis
   📥 Args: {
      "action": "pylint",
      "file_path": "mybot/providers/base.py"
   }
   🔄 Auto-extracted file path from step 1: mybot/providers/base.py
   ✅ Success
   📤 Result preview: {
     "score": 8.5,
     "errors": 2,
     ...
   }
```

**Parallel Execution**:

When multiple steps are ready (no dependencies), they execute in parallel:

```
🔄 Executing 2 step(s) in parallel...

📍 Step 2: Analyze code quality
📍 Step 3: Count lines of code
   (Both execute simultaneously)
```

</details>

<details>
<summary><b>Click to expand: Phase 3 - Synthesis</b></summary>

**How It Works**:

The synthesis phase creates a natural language summary of the execution results:

```python
async def _synthesize(
    self,
    user_message: str,
    plan: list,
    results: dict,
    stream: bool,
    stream_callback
) -> str:
    """Create final response from execution results."""
    # 1. Build execution summary
    # 2. Send to LLM with results
    # 3. Generate natural language response
    # 4. Support streaming output
```

**Synthesis Process**:

1. **Summary Building**: Creates a structured summary of all step results
2. **LLM Synthesis**: Sends summary to LLM with instructions to create a helpful response
3. **Streaming Support**: Supports streaming for real-time output
4. **Error Highlighting**: Mentions any errors or issues that occurred

**Example Output**:

```
📊 PHASE 3: Synthesizing Results...
============================================================

💬 Final Response:
------------------------------------------------------------
**What was done**

1. **Search** – Found all TODO comments in the `mybot/` directory.
   *Result:* 5 TODO comments found across 3 files.

2. **Code Analysis (pylint)** – Ran pylint on files with TODOs.
   *Result:* Code quality score: 8.5/10. Found 2 errors and 5 warnings.

3. **Coverage Report** – Generated test coverage report.
   *Result:* Coverage: 85% (1200/1412 lines covered).

**Key Findings**
- Most TODOs are in `mybot/providers/base.py`
- Pylint found 2 critical errors that should be addressed
- Test coverage is good but could be improved for edge cases
```

</details>

<details>
<summary><b>Click to expand: Advanced Features</b></summary>

**1. Dynamic Argument Resolution**

The system automatically extracts file paths and data from previous step results:

```python
# Example: Step 1 finds files, Step 2 processes them
Step 1: find_files(pattern="*.py", path="tests")
  → Returns: ["test1.py", "test2.py"]

Step 2: pylint(file_path="{{step1.output[0]}}")
  → Automatically resolves to: "tests/test1.py"
```

**2. Template Placeholders**

Steps can reference previous step outputs using template syntax:

- `{{step1.output}}` - Full output from step 1
- `{{step1.output[0]}}` - First item from step 1's output
- `{{step2.output[1]}}` - Second item from step 2's output

**3. Error Handling**

- Failed dependencies are detected before extraction
- Steps with failed dependencies skip extraction
- Tool-specific error detection (distinguishes errors from legitimate outputs)
- Retry logic with configurable attempts

**4. Plan Validation**

Plans are validated before execution:
- Step numbers must be valid integers
- No duplicate step numbers
- All tools must exist in registry
- Dependencies must reference previous steps only
- Required parameters must be present

**5. Parallel Execution**

Independent steps (no dependencies) execute in parallel:
- Significantly faster for multi-step tasks
- Automatic dependency resolution
- Safe concurrent execution

</details>

<details>
<summary><b>Click to expand: Example Usage</b></summary>

**Example 1: Simple Multi-Step Task**

```
User: Find all Python files in tests directory, run pylint on them, then generate a coverage report

Agent:
============================================================
🧠 PLANNING MODE ACTIVATED
============================================================

📋 PHASE 1: Creating Execution Plan...
   Generated 3 step(s)

✅ Plan created with 3 step(s)

📝 Execution Plan:
   Step 1: Find all Python files in tests directory
   Step 2: Run pylint on the first file (depends on: [1])
   Step 3: Generate coverage report (depends on: [2])

⚙️  PHASE 2: Executing Plan...
📍 Step 1: Find all Python files...
   ✅ Success: Found 5 files

📍 Step 2: Run pylint...
   🔄 Auto-extracted file path: tests/test_agent.py
   ✅ Success: Score 8.5/10

📍 Step 3: Generate coverage report...
   ✅ Success: Coverage 85%

📊 PHASE 3: Synthesizing Results...
💬 Final Response:
Found 5 Python test files. Ran pylint on tests/test_agent.py 
with score 8.5/10. Coverage report shows 85% coverage...
```

**Example 2: Complex Workflow with Dependencies**

```
User: Find all TODOs in mybot directory, then run pylint on files that have TODOs, then format those files with black

Agent:
📋 PHASE 1: Creating Execution Plan...
   Generated 3 step(s)

📝 Execution Plan:
   Step 1: Find all TODOs in mybot directory
   Step 2: Run pylint on files with TODOs (depends on: [1])
   Step 3: Format files with black (depends on: [2])

⚙️  PHASE 2: Executing Plan...
📍 Step 1: Find all TODOs...
   ✅ Success: Found 5 TODOs in 3 files

📍 Step 2: Run pylint...
   🔄 Auto-extracted file path: mybot/providers/base.py
   ✅ Success: Found 2 errors

📍 Step 3: Format with black...
   🔄 Auto-extracted file path: mybot/providers/base.py
   ✅ Success: Formatted successfully
```

**Example 3: Parallel Execution**

```
User: Count lines of code in mybot, find all test files, and search for "async def" patterns

Agent:
📋 PHASE 1: Creating Execution Plan...
   Generated 3 step(s)

📝 Execution Plan:
   Step 1: Count lines of code in mybot
   Step 2: Find all test files
   Step 3: Search for "async def" patterns

⚙️  PHASE 2: Executing Plan...
🔄 Executing 3 step(s) in parallel...

📍 Step 1: Count lines...
📍 Step 2: Find test files...
📍 Step 3: Search patterns...
   (All execute simultaneously)

✅ All steps completed successfully
```

</details>

<details>
<summary><b>Click to expand: Best Practices</b></summary>

**1. Use Planning for Complex Tasks**

Planning mode is ideal for:
- ✅ Multi-step workflows
- ✅ Tasks with dependencies
- ✅ Data flow between steps
- ✅ Complex analysis tasks

Standard mode is better for:
- ✅ Simple single-step tasks
- ✅ Direct tool calls
- ✅ Conversational responses

**2. Clear Task Descriptions**

Provide clear, specific task descriptions:
- ✅ "Find all TODOs, run pylint on those files, then format them"
- ❌ "Do stuff with code"

**3. Leverage Dependencies**

Use dependencies to ensure correct execution order:
```json
{
  "step": 2,
  "depends_on": [1]  // Ensures step 1 completes first
}
```

**4. Use Template Placeholders**

Reference previous step outputs explicitly:
```json
{
  "args": {
    "file_path": "{{step1.output[0]}}"
  }
}
```

**5. Monitor Execution**

Use `verbose=True` to see detailed execution logs:
- Step execution status
- Argument resolution
- Error details
- Parallel execution indicators

</details>

<details>
<summary><b>Click to expand: Troubleshooting</b></summary>

**Issue: Plan validation fails**

**Solution**: The system auto-fixes common issues, but check:
- Tool names match exactly (case-sensitive)
- Required parameters are included
- Dependencies reference valid step numbers

**Issue: Steps not executing in parallel**

**Possible Causes**:
- Steps have dependencies (must execute sequentially)
- Only one step is ready at a time
- Dependencies not properly defined

**Solution**: Review dependency structure. Independent steps will execute in parallel automatically.

**Issue: Argument resolution fails**

**Possible Causes**:
- Previous step failed
- Output format doesn't match expected pattern
- Template placeholder syntax incorrect

**Solution**:
- Check previous step results
- Use explicit template placeholders: `{{step1.output[0]}}`
- Verify output format from previous step

**Issue: Retry logic not working**

**Check**:
- `max_retries` is set > 0
- Error is retryable (not validation error)
- Tool supports retries

</details>

---

**Please give me a STAR 🌟 if you find this useful!**
