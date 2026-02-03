# Building nanobot from Scratch: Implementation Guide

This guide walks you through building a small bot AI assistant step-by-step, in the correct order.

## 🎯 Overview

**Goal**: Build a lightweight AI assistant that can:
- Chat with users via CLI
- Execute tools (read files, run commands)
- Maintain conversation history
- Use LLM providers for reasoning

**Timeline**: ~2-3 weeks for a basic version, ~1-2 months for full feature set

---

## 📋 Prerequisites

- Python 3.11+
- Basic async/await knowledge
- Understanding of LLM APIs
- Familiarity with type hints

---

## 🏗️ Implementation Phases

### Phase 0: Setup & Planning

**Goal**: Set up project structure and understand requirements

**Tasks**:
1. Create project structure
2. Plan architecture

**Project Structure**:
```
edubot/
├── mybot/
│   ├── __init__.py
│   ├── agent/
│   ├── providers/
│   ├── tools/
│   └── session/
├── tests/
├── pyproject.toml
└── README.md
```

**Deliverable**: Project skeleton ready


---

### Phase 1: Core Data Models

**Goal**: Define the basic data structures

**Why First**: Everything else depends on these

**Files to Create**:
1. `mybot/models.py` - Core data models
2. `mybot/exceptions.py` - Custom exceptions

**Implementation**:

```python
# mybot/models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any

@dataclass
class Message:
    """A conversation message."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ToolCall:
    """A tool call request."""
    id: str
    name: str
    args: dict[str, Any]

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str | None
    tool_calls: list[ToolCall] = None
    finish_reason: str = "stop"
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
```

```python
# mybot/exceptions.py
class BotException(Exception):
    """Base exception."""
    pass

class LLMError(BotException):
    """LLM provider error."""
    pass

class ToolError(BotException):
    """Tool execution error."""
    pass
```

**Test**:
```python
# tests/test_models.py
def test_message_creation():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.timestamp is not None
```

**Deliverable**: Core models defined and tested

---

### Phase 2: LLM Provider

**Goal**: Connect to an LLM API

**Why Second**: Agent needs LLM to function

**Files to Create**:
1. `mybot/providers/base.py` - Abstract interface
2. `mybot/providers/openrouter_provider.py` - OpenRouter implementation (recommended)
   OR `mybot/providers/openai_provider.py` - OpenAI implementation

**Important**: nanobot uses **dict-based messages**, not Message objects. Messages are `list[dict[str, Any]]` where each dict has `{"role": "user", "content": "..."}`.

**Implementation**:

```python
# mybot/providers/base.py
from abc import ABC, abstractmethod
from typing import Any
from mybot.models import LLMResponse

class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],  # Note: dicts, not Message objects!
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier.
        """
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
    """Provider using OpenRouter API (supports many models including free ones)."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/yourusername/mybot",  # Optional
            }
        )
    
    async def chat(
        self,
        messages: list[dict[str, Any]],  # Already in dict format!
        tools: list[dict[str, Any]] | None = None,
        model: str = "nvidia/nemotron-3-nano-30b-a3b:free",  # Free model
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": messages,  # Use directly, no conversion needed
            "temperature": 0.7,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Parse response
        choice = data["choices"][0]
        message = choice["message"]
        
        tool_calls = []
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                args = tc["function"]["arguments"]
                # Arguments come as JSON string, parse if needed
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args
                ))
        
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop")
        )
```

**Test**:

1. **Set up API Key**:
   
   Create a `.env` file in the project root directory:
   ```bash
   # Create .env file
   touch .env
   ```
   
   Add your OpenRouter API key to the `.env` file:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
   ```
   
   **Note**: 
   - Get your API key from https://openrouter.ai/keys
   - API keys start with `sk-or-v1-`
   - You can use quotes or no quotes: both formats work
   - The `.env` file should be in the project root (same directory as `pyproject.toml`)

2. **Install Dependencies**:
   
   Make sure you have `python-dotenv` installed to load the `.env` file:
   ```bash
   pip install python-dotenv
   ```

3. **Run the Test**:
   
   ```bash
   # Activate your conda environment (if using conda)
   conda activate vis-py311
   
   # Run the provider test
   python tests/test_provider.py
   ```
   
   The test will:
   - Test basic chat functionality with a free model
   - Test tool calling functionality
   - Print responses and tool calls for verification
   
   **Expected Output**:
   ```
   Testing OpenRouter basic chat...
   Response: Hello, World!
   ✓ Test passed
   
   Testing OpenRouter with tools...
   Tool call: get_weather with args: {'location': 'San Francisco, CA'}
   ✓ Test passed
   
   All tests passed!
   ```

4. **Troubleshooting**:
   
   - **404 Error**: Model not found - check that the model name is correct
   - **401/403 Error**: Authentication failed - verify your API key is correct
   - **402 Error**: Insufficient credits - some models require API credits
   - **ModuleNotFoundError**: Make sure you're running from the project root and dependencies are installed

---

### Phase 3: Tool System

**Goal**: Create extensible tool system

**Why Third**: Agent needs tools to interact with environment

**Files to Create**:
1. `mybot/tools/base.py` - Tool interface
2. `mybot/tools/registry.py` - Tool registry
3. `mybot/tools/filesystem.py` - File tools
4. `mybot/tools/shell.py` - Shell tool

**Implementation**:

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
        """JSON Schema for parameters."""
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
    
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)
    
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
```
```python
# mybot/tools/shell.py

import asyncio
import os
from typing import Any
from mybot.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""
    
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
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                }
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
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"Error: Command timed out after {self.timeout} seconds"
            
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
            
            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"
            
            return result
            
        except Exception as e:
            return f"Error executing command: {str(e)}"
```

**Test**:
```python
# test/test_tools.py
from mybot.tools.registry import ToolRegistry
from mybot.tools.filesystem import ReadFileTool
from mybot.tools.shell import ExecTool
import asyncio

async def test_tool_registry():
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ExecTool())
    assert registry.get("read_file") is not None
    assert registry.get("exec") is not None
    assert len(registry.get_definitions()) == 2


async def test_read_file():
    tool = ReadFileTool()
    # Create test file
    test_file = Path("/tmp/test.txt")
    test_file.write_text("Hello, World!\nThis test file is for testing the read file tool.")

    result = await tool.execute(path=str(test_file))
    print(result)
    assert "Hello, World!" in result
    print("✓ Test [read file] passed")
async def test_exec():
    tool = ExecTool()
    result = await tool.execute(command="echo 'Hello, World!'")
    print(result)
    assert "Hello, World!" in result
    print("✓ Test [exec] passed")
```

### Phase 4: Session Management 

**Goal**: Persist conversation history

**Why Fourth**: Agent needs context from previous messages

**Files to Create**:
1. `mybot/session/manager.py` - Session management

**Implementation**:

```python
# mybot/session/manager.py
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
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent
        ]
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

**Test**:
```python
# tests/test_session.py
import tempfile
from pathlib import Path
from mybot.session.manager import SessionManager

def test_session_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(Path(tmpdir))
        session = manager.get_or_create("test:123")
        session.add_message("user", "Hello")
        manager.save(session)
        
        # Reload
        session2 = manager.get_or_create("test:123")
        assert len(session2.messages) == 1
```

**Deliverable**: Sessions persist and load correctly

---