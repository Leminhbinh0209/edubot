# Building nanobot from Scratch: Implementation Guide

This guide walks you through building a nanobot-like AI assistant step-by-step, in the correct order.

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

### Phase 0: Setup & Planning (Day 1)

**Goal**: Set up project structure and understand requirements

**Tasks**:
1. Create project structure
2. Set up `pyproject.toml` with dependencies
3. Create basic README
4. Plan architecture

**Project Structure**:
```
mybot/
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

**Dependencies to Add**:
```toml
[project]
dependencies = [
    "httpx>=0.25.0",
    "pydantic>=2.0.0",
    "loguru>=0.7.0",
    "litellm>=1.0.0",  # Optional: for multi-provider support (like nanobot)
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

**Note**: You can use either:
- Direct HTTP client (`httpx`) for simple providers
- LiteLLM for multi-provider support (recommended, like nanobot)

**Deliverable**: Project skeleton ready


---

### Phase 1: Core Data Models (Day 1-2)

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

### Phase 2: LLM Provider (Day 2-3)

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
