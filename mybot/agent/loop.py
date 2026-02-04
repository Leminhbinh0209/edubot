import asyncio
from pathlib import Path
from typing import Callable, Optional, Awaitable, Union
from mybot.providers.base import LLMProvider
from mybot.tools.registry import ToolRegistry
from mybot.session.manager import SessionManager
from mybot.models import Message

class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        model: str = "nvidia/nemotron-3-nano-30b-a3b:free",
        tools: ToolRegistry = None,
        sessions: SessionManager = None,
        max_iterations: int = 10,):
        if tools is None:
            tools = ToolRegistry()
        if sessions is None:
            sessions = SessionManager(Path.home() / ".mybot" / "sessions")
        self.provider = provider
        self.tools = tools
        self.sessions = sessions
        self.max_iterations = max_iterations
        self.model = model
    async def process_message(
        self,
        user_message: str,
        session_key: str = "default",
        stream: bool = False,
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None,
    ) -> str:
        """Process a user message and return response."""
        import json
        session = self.sessions.get_or_create(session_key)
        messages = []
        system_prompt = """You are a helpful AI assistant. You have access to tools.
        When you need to use a tool, call it. Otherwise, respond directly to the user."""
        messages.append({"role": "system", "content": system_prompt})
        history = session.get_history()
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        iteration = 0
        final_response = None
        # print(f"\n[Agent Loop] Processing message: {user_message[:50]}...")
        # print(f"[Agent Loop] Session: {session_key}, Max iterations: {self.max_iterations}")
        
        while iteration < self.max_iterations:
            iteration += 1
            # print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            # print(f"[LLM Request] Sending {len(messages)} messages to model: {self.model}")
            # print(f"=== START Messages ===\n{json.dumps(messages, indent=2)}\n === END Messages ===")
            response = await self.provider.chat(
                messages=messages, 
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # print(f"[LLM Response] Content: {response.content[:100] if response.content else '(empty)'}...")
            # print(f"[LLM Response] Has tool calls: {response.has_tool_calls()}")
            # print(f"[LLM Response] Finish reason: {response.finish_reason}")
            
            if response.has_tool_calls():
                # print(f"[Tool Calls] {len(response.tool_calls)} tool call(s) detected:")
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
                
                for i, tool_call in enumerate(response.tool_calls, 1):
                    print(f"  [{i}] Tool: {tool_call.name}")
                    print(f"      Args: {tool_call.args}")
                
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": tool_call_dicts
                })
                
                # print(f"[Tool Execution] Executing {len(response.tool_calls)} tool(s)...")
                for i, tool_call in enumerate(response.tool_calls, 1):
                    print(f"  [{i}] Executing: {tool_call.name}({tool_call.args})")
                    result = await self.tools.execute(tool_call.name, tool_call.args)
                    result_preview = result[:200] if len(result) > 200 else result
                    print(f"      Result: {result_preview}{'...' if len(result) > 200 else ''}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result
                    })
                # print(f"[Tool Execution] All tools executed, continuing loop...")
            else:
                # print(f"[Final Response] No tool calls, returning response")
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
                break
        if final_response is None:
            final_response = "I'm having trouble processing that request."
        session.add_message("user", user_message)
        session.add_message("assistant", final_response)
        self.sessions.save(session)
        return final_response