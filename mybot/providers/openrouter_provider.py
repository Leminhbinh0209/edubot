import httpx
import json
from typing import Any, AsyncIterator
from mybot.providers.base import LLMProvider
from mybot.models import LLMResponse, ToolCall

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/yourusername/mybot",  # Optional
                "Content-Type": "application/json",
            },
            timeout=60.0
        )
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, model: str | None = None) -> LLMResponse:
        payload = {
            "model": model,
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
                # Arguments come as JSON string, parse if needed
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
        return LLMResponse(content=message["content"], tool_calls=tool_calls, finish_reason=choice["finish_reason"])
    
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
        # For now, we'll only stream when there are no tools
        # Tool calls require the full response to parse tool_calls
        if tools:
            # Fall back to non-streaming for tool calls
            response = await self.chat(messages, tools, model)
            if response.content:
                yield response.content
            return
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "stream": True,
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