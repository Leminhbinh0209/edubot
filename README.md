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