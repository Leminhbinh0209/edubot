from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime=None
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ToolCall:
    "Tool call is a call to a tool"
    id: str
    name: str
    args: Dict[str, Any]
@dataclass
class LLMResponse:
    content: str | None
    tool_calls: List[ToolCall] = None
    finish_reason: str = "stop"
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
