
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