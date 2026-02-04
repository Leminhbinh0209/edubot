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

- Add more tools (web search, database queries, APIs)
- Improve error handling and retry logic
- Add streaming responses
- Build a web interface
- Add multi-modal capabilities

Please switch to `advance_dev` for more feature.
---

**Please give me a STAR 🌟 if you find this useful!**
