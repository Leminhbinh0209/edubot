
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
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