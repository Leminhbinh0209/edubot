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
        max_iterations: int = 10,
        use_planning: bool = False,
        verbose: bool = True,
        max_retries: int = 2):
        if tools is None:
            tools = ToolRegistry()
        if sessions is None:
            sessions = SessionManager(Path.home() / ".mybot" / "sessions")
        self.provider = provider
        self.tools = tools
        self.sessions = sessions
        self.max_iterations = max_iterations
        self.model = model
        self.use_planning = use_planning
        self.verbose = verbose
        self.max_retries = max_retries
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
        
        # Use planning mode if enabled
        if self.use_planning:
            return await self._process_with_planning(user_message, session_key, stream, stream_callback)
        
        # Original non-planning mode
        messages = []
        system_prompt = """You are a helpful AI assistant with access to powerful tools. 

IMPORTANT: You MUST use tools when the user requests:
- Code formatting (black, formatting) → use code_analysis tool with action="black_format"
- Code linting (pylint, linting) → use code_analysis tool with action="pylint"
- Type checking (mypy, types) → use code_analysis tool with action="mypy_check"
- Searching files/patterns (grep, find, search) → use search tool
- Finding TODOs/comments → use search tool with action="find_todos"
- Running tests (pytest, jest, unittest) → use test_runner tool
- Coverage reports → use test_runner tool with action="coverage_report"
- Reading files → use read_file tool
- Executing commands → use exec tool

When a user asks you to perform any of these actions, you MUST call the appropriate tool. Do not try to do these tasks yourself - always use the tools.

Examples:
- User: "Format mybot/cli.py with black" → Call code_analysis(action="black_format", file_path="mybot/cli.py")
- User: "Find all TODOs" → Call search(action="find_todos", path=".")
- User: "Run pytest tests" → Call test_runner(action="run_pytest", path="./tests")

Only respond directly without tools for general conversation, explanations, or questions that don't require tool execution."""
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
    
    async def _plan(self, user_message: str, session) -> list[dict]:
        """Create execution plan."""
        import json
        
        # Build tools description
        tool_definitions = self.tools.get_definitions()
        tools_description = "\n".join([
            f"- {tool['function']['name']}: {tool['function'].get('description', 'No description')}"
            for tool in tool_definitions
        ])
        
        # Build detailed tool schemas for planning
        tool_schemas = []
        for tool_def in tool_definitions:
            tool_name = tool_def['function']['name']
            params = tool_def['function'].get('parameters', {})
            required = params.get('required', [])
            properties = params.get('properties', {})
            
            schema_desc = f"\n{tool_name}:"
            schema_desc += f"\n  Description: {tool_def['function'].get('description', 'No description')}"
            schema_desc += f"\n  Required parameters: {', '.join(required) if required else 'None'}"
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', '')
                enum_vals = param_info.get('enum', [])
                if enum_vals:
                    schema_desc += f"\n    - {param_name} ({param_type}): {param_desc} - Must be one of: {enum_vals}"
                else:
                    schema_desc += f"\n    - {param_name} ({param_type}): {param_desc}"
            
            tool_schemas.append(schema_desc)
        
        tools_schema_description = "\n".join(tool_schemas)
        
        planning_messages = [
            {"role": "system", "content": f"""You are a planning assistant. Create detailed, step-by-step execution plans.

Available tools with their schemas:
{tools_schema_description}

For each step, specify:
1. step: step number (1, 2, 3, ...)
2. tool: exact tool name (must match one of the available tools)
3. args: arguments dict (MUST include ALL required parameters from the tool schema)
4. reasoning: why this step is needed
5. depends_on: list of previous step numbers this depends on (empty list if no dependencies)

CRITICAL RULES:
- You MUST include ALL required parameters in the args dict
- Check the "Required parameters" list for each tool and include every one
- When a step depends on another, extract specific information from the previous step's results
- For find_files: MUST include "pattern" parameter (e.g., "*.py")
- Use specific file paths (ending in .py) not directories when tools require file paths
- If you need to process multiple files from a previous step, create separate steps for each file or use a pattern that works with the tool"""},
            {"role": "user", "content": f"""Create a detailed execution plan for: {user_message}

CRITICAL: When a step depends on previous steps that return file paths or lists:
- Extract the specific file paths from the previous step's output
- Use those exact file paths in the args (e.g., "file_path": "mybot/file.py" not "mybot")
- If the previous step returns a list of files, you may need to process each file separately

Respond ONLY with valid JSON in this exact format:
{{
  "steps": [
    {{
      "step": 1,
      "tool": "tool_name",
      "args": {{"key": "value"}},
      "reasoning": "Why this step is needed",
      "depends_on": []
    }},
    {{
      "step": 2,
      "tool": "tool_name",
      "args": {{"key": "specific_value_from_step_1"}},
      "reasoning": "Why this step is needed and how it uses step 1 results",
      "depends_on": [1]
    }}
  ]
}}

Make sure tool names exactly match available tools and args match their schemas. Use specific file paths, not directories."""}
        ]
        
        try:
            response = await self.provider.chat(messages=planning_messages, model=self.model, tools=None)
            
            # Extract JSON robustly
            plan_data = self._extract_json(response.content)
            steps = plan_data.get("steps", [])
            
            # Validate plan before returning
            validation_errors = self._validate_plan(steps)
            if validation_errors:
                if self.verbose:
                    print(f"   ⚠️  Plan validation failed:")
                    for error in validation_errors:
                        print(f"      - {error}")
                
                # Try to fix common issues
                steps = self._fix_plan(steps, validation_errors)
            
            if self.verbose:
                print(f"   Generated {len(steps)} step(s)")
            return steps
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  Error creating plan: {e}")
            return []
    
    def _extract_json(self, content: str) -> dict:
        """Robustly extract JSON from LLM response."""
        import json
        import re
        
        content = content.strip()
        
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Find all code blocks
        code_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        
        # Try each code block
        for block in code_blocks:
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON object directly (last resort)
        json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not extract valid JSON from response")
    
    def _validate_plan(self, steps: list[dict]) -> list[str]:
        """Validate plan structure and dependencies."""
        errors = []
        seen_steps = set()
        
        for step in steps:
            step_num = step.get('step')
            
            # Check step number
            if not step_num or not isinstance(step_num, int):
                errors.append(f"Invalid step number: {step_num}")
                continue
            
            if step_num in seen_steps:
                errors.append(f"Duplicate step number: {step_num}")
            seen_steps.add(step_num)
            
            # Check tool exists
            tool_name = step.get('tool')
            if not tool_name:
                errors.append(f"Step {step_num}: Missing tool name")
            elif not self.tools.get(tool_name):
                errors.append(f"Step {step_num}: Tool '{tool_name}' not found")
            
            # Check dependencies exist
            depends_on = step.get('depends_on', [])
            for dep in depends_on:
                if dep >= step_num:
                    errors.append(f"Step {step_num}: Invalid dependency on future step {dep}")
        
        return errors
    
    def _fix_plan(self, steps: list[dict], errors: list[str]) -> list[dict]:
        """Attempt to fix common plan issues."""
        # Renumber steps sequentially
        for i, step in enumerate(steps, 1):
            step['step'] = i
        
        # Remove steps with invalid tools
        valid_steps = []
        for step in steps:
            if self.tools.get(step.get('tool')):
                valid_steps.append(step)
            elif self.verbose:
                print(f"   🔧 Removed step with invalid tool: {step.get('tool')}")
        
        return valid_steps
    
    async def _execute_plan(self, plan: list[dict]) -> dict:
        """Execute plan steps, parallelizing independent steps."""
        import json
        results = {}
        completed_steps = set()
        remaining_steps = sorted(plan, key=lambda x: x.get('step', 0))
        
        while remaining_steps:
            # Find steps ready to execute (dependencies met)
            ready_steps = []
            for step in remaining_steps:
                depends_on = step.get('depends_on', [])
                if all(dep in completed_steps for dep in depends_on):
                    ready_steps.append(step)
            
            if not ready_steps:
                if self.verbose:
                    print("⚠️  No steps ready to execute, breaking to avoid infinite loop")
                break
            
            # Execute ready steps in parallel
            if len(ready_steps) > 1 and self.verbose:
                print(f"\n🔄 Executing {len(ready_steps)} step(s) in parallel...")
            
            tasks = [self._execute_single_step(step, results) for step in ready_steps]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, step_results):
                step_num = step['step']
                if isinstance(result, Exception):
                    results[step_num] = {
                        "success": False,
                        "error": str(result),
                        "step": step
                    }
                else:
                    results[step_num] = result
                
                completed_steps.add(step_num)
                remaining_steps.remove(step)
        
        return results
    
    async def _execute_single_step(self, step: dict, results: dict) -> dict:
        """Execute a single step and return result."""
        import json
        step_num = step.get('step', 0)
        tool_name = step.get('tool', '')
        args = step.get('args', {})
        reasoning = step.get('reasoning', 'No reasoning provided')
        depends_on = step.get('depends_on', [])
        
        # Check dependencies
        if depends_on:
            missing_deps = [d for d in depends_on if d not in results]
            if missing_deps:
                return {
                    "success": False,
                    "error": f"Dependencies not met: {missing_deps}",
                    "step": step
                }
        
        # Resolve dynamic arguments from previous step results
        resolved_args = self._resolve_args_from_results(args, results, depends_on, tool_name)
        
        # Validate required parameters before execution
        validation_error = self._validate_step_args(tool_name, resolved_args)
        if validation_error:
            if self.verbose:
                print(f"\n📍 Step {step_num}: {reasoning}")
                print(f"   🔧 Tool: {tool_name}")
                print(f"   📥 Args: {json.dumps(resolved_args, indent=6)}")
                print(f"   ⚠️  Validation Error: {validation_error}")
            return {
                "success": False,
                "error": validation_error,
                "step": step
            }
        
        if self.verbose:
            print(f"\n📍 Step {step_num}: {reasoning}")
            print(f"   🔧 Tool: {tool_name}")
            print(f"   📥 Args: {json.dumps(resolved_args, indent=6)}")
        
        # Execute with retry logic
        return await self._execute_single_step_with_retry(step, resolved_args, results)
    
    async def _execute_single_step_with_retry(self, step: dict, resolved_args: dict, results: dict) -> dict:
        """Execute step with retry on failure."""
        step_num = step.get('step', 0)
        tool_name = step.get('tool', '')
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.tools.execute(tool_name, resolved_args)
                
                # Check if result indicates an error
                result_str = str(result) if result else ""
                is_error = self._is_error_result(result_str, tool_name)
                
                if is_error:
                    result_dict = {
                        "success": False,
                        "error": result_str[:500],  # Limit error message length
                        "output": result_str,
                        "step": step
                    }
                    
                    if attempt < self.max_retries:
                        if self.verbose:
                            print(f"   🔄 Retry {attempt + 1}/{self.max_retries}...")
                        await asyncio.sleep(1)  # Brief delay
                        continue
                    
                    if self.verbose:
                        print(f"   ❌ Error: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")
                    return result_dict
                else:
                    result_dict = {
                        "success": True,
                        "output": result,
                        "step": step
                    }
                    
                    if self.verbose:
                        # Show preview of result
                        result_preview = result[:150] if len(result) > 150 else result
                        result_lines = result_preview.split('\n')
                        if len(result_lines) > 3:
                            result_preview = '\n'.join(result_lines[:3]) + f"\n   ... ({len(result_lines) - 3} more lines)"
                        
                        print(f"   ✅ Success")
                        print(f"   📤 Result preview: {result_preview}")
                    
                    return result_dict
                    
            except Exception as e:
                if attempt < self.max_retries:
                    if self.verbose:
                        print(f"   🔄 Retry {attempt + 1}/{self.max_retries} after exception...")
                    await asyncio.sleep(1)
                    continue
                
                if self.verbose:
                    print(f"   ❌ Error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "step": step
                }
        
        # Should never reach here, but just in case
        return {
            "success": False,
            "error": "Max retries exceeded",
            "step": step
        }
    
    def _validate_step_args(self, tool_name: str, args: dict) -> str | None:
        """Validate that all required parameters are present in args."""
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found"
        
        # Get tool schema
        schema = tool.to_schema()
        params = schema.get('function', {}).get('parameters', {})
        required = params.get('required', [])
        
        # Check if all required parameters are present
        missing = [param for param in required if param not in args]
        if missing:
            return f"Missing required parameters: {', '.join(missing)}. Required: {', '.join(required)}"
        
        return None
    
    def _is_error_result(self, result: str, tool_name: str) -> bool:
        """Determine if tool result indicates an error."""
        result_lower = result.lower()
        
        # Definite error indicators
        error_indicators = [
            result.startswith("Error:"),
            result.startswith("ERROR:"),
            result.startswith("Exception:"),
            "traceback (most recent call last)" in result_lower,
            "file not found:" in result_lower,
        ]
        
        if any(error_indicators):
            return True
        
        # Tool-specific error detection
        if tool_name == "read_file":
            return "does not exist" in result_lower or "permission denied" in result_lower
        
        if tool_name == "exec":
            return "command not found" in result_lower or "exit code" in result_lower
        
        # "not found" is often legitimate (search results, etc)
        # Only treat as error if it's about the tool itself
        if "tool not found" in result_lower or "method not found" in result_lower:
            return True
        
        return False
    
    def _should_extract_file_path(self, tool_name: str, arg_name: str, arg_value: str) -> bool:
        """Determine if we should extract file path from previous results."""
        from pathlib import Path
        
        # Tools that accept directories, not just files
        DIRECTORY_TOOLS = {'search', 'test_runner'}
        DIRECTORY_ACTIONS = {'find_files', 'grep', 'find_todos', 'count_lines', 'run_pytest', 'run_jest', 'coverage_report'}
        
        # Check if this is a directory-accepting tool/action
        if tool_name in DIRECTORY_TOOLS:
            if arg_name == 'path':
                return False  # These tools work with directories, don't extract
        
        # Check if action is directory-based
        if isinstance(arg_value, dict):
            action = arg_value.get('action', '')
            if action in DIRECTORY_ACTIONS and arg_name == 'path':
                return False
        
        # If it's explicitly a file_path parameter, extract if needed
        if arg_name == 'file_path':
            # Check for template placeholders
            if '{{step' in str(arg_value):
                return True
            # Check if file doesn't exist and looks like placeholder
            if str(arg_value).endswith('.py'):
                if not Path(str(arg_value)).exists():
                    placeholder_patterns = ['example', 'todo_example', 'test_file', 'sample', 'placeholder']
                    return any(pattern in str(arg_value).lower() for pattern in placeholder_patterns)
        
        return False
    
    def _resolve_args_from_results(self, args: dict, results: dict, depends_on: list, tool_name: str = '') -> dict:
        """Resolve dynamic arguments by extracting file paths from previous step results.
        
        Handles:
        - Template placeholders like {{step1.output[0]}}
        - Missing file paths that need extraction
        - Formatted output strings with file lists
        """
        import json
        import re
        from pathlib import Path
        
        resolved_args = args.copy()
        
        # Check for failed dependencies first
        if depends_on:
            failed_deps = [d for d in depends_on if d in results and not results[d].get('success')]
            if failed_deps:
                if self.verbose:
                    print(f"   ⚠️  Warning: Dependencies {failed_deps} failed, skipping extraction")
                return resolved_args  # Don't try to resolve from failed steps
        
        # Check each argument to see if we should extract
        for arg_name, arg_value in resolved_args.items():
            if not self._should_extract_file_path(tool_name, arg_name, arg_value):
                continue
            
            file_path = str(arg_value)
            
            # Check if it's a template placeholder (e.g., {{step1.output[0]}})
            template_pattern = r'\{\{step(\d+)(?:\.output(?:\[(\d+)\])?)?\}\}'
            template_match = re.search(template_pattern, file_path)
            
            extract_from_step = None
            extract_index = None
            
            if template_match:
                # It's a template placeholder - extract from specified step
                extract_from_step = int(template_match.group(1))
                if template_match.group(2):
                    extract_index = int(template_match.group(2))
            
                # Determine which step to extract from
                steps_to_check = [extract_from_step] if extract_from_step else depends_on
                
                for dep_step in steps_to_check:
                    if dep_step in results:
                        dep_result = results[dep_step]
                        if dep_result.get('success'):
                            output = dep_result.get('output', '')
                            
                            # Try multiple extraction methods
                            extracted_path = None
                            
                            try:
                                # Method 1: Extract from formatted string like "Found 5 results:\ntest_agent.py\ntest_models.py"
                                # Look for lines that are just filenames ending in .py
                                output_str = str(output)
                                lines = output_str.split('\n')
                                
                                # Collect all .py files found
                                py_files = []
                                for line in lines:
                                    line = line.strip()
                                    # Skip header lines
                                    if line.startswith('Found') or line.startswith('results:') or line.startswith('No results') or not line:
                                        continue
                                    # Check if it's a filename (ends in .py, might have path)
                                    if line.endswith('.py'):
                                        py_files.append(line)
                                
                                # Select the file based on index (if specified) or use first one
                                if py_files:
                                    selected_file = py_files[extract_index] if extract_index is not None and extract_index < len(py_files) else py_files[0]
                                    
                                    # If it's just a filename, try to construct full path from previous step's path
                                    if '/' not in selected_file and '\\' not in selected_file:
                                        # Get the path from the previous step's args
                                        prev_step_info = results[dep_step].get('step', {})
                                        prev_args = prev_step_info.get('args', {}) if isinstance(prev_step_info, dict) else {}
                                        prev_path = prev_args.get('path', '')
                                        
                                        if prev_path:
                                            # Construct full path
                                            extracted_path = str(Path(prev_path) / selected_file)
                                        else:
                                            extracted_path = selected_file
                                    else:
                                        extracted_path = selected_file
                                
                                # Method 2: Extract from dict format like {'file': 'path/to/file.py'}
                                if not extracted_path:
                                    file_pattern = r"'file':\s*'([^']+\.py)'"
                                    matches = re.findall(file_pattern, output)
                                    if matches:
                                        extracted_path = matches[0]
                                
                                # Method 3: Try JSON parsing
                                if not extracted_path:
                                    json_match = re.search(r'\[.*?\]', output, re.DOTALL)
                                    if json_match:
                                        try:
                                            parsed = json.loads(json_match.group(0))
                                            if isinstance(parsed, list) and len(parsed) > 0:
                                                first_item = parsed[0]
                                                if isinstance(first_item, dict) and 'file' in first_item:
                                                    extracted_path = first_item['file']
                                                elif isinstance(first_item, str) and first_item.endswith('.py'):
                                                    extracted_path = first_item
                                        except json.JSONDecodeError:
                                            pass
                                
                                # Method 4: Look for quoted file paths
                                if not extracted_path:
                                    path_pattern = r'["\']([^"\']+\.py)["\']'
                                    path_matches = re.findall(path_pattern, output)
                                    if path_matches:
                                        valid_paths = [p for p in path_matches if '/' in p or '\\' in p]
                                        if valid_paths:
                                            extracted_path = valid_paths[0]
                                
                                # If we found a path, use it
                                if extracted_path:
                                    resolved_args[arg_name] = extracted_path
                                    if self.verbose:
                                        print(f"   🔄 Auto-extracted {arg_name} from step {dep_step}: {extracted_path}")
                                    break
                                    
                            except Exception as e:
                                # Silently continue to next dependency
                                pass
        
        return resolved_args
    
    async def _synthesize(
        self,
        user_message: str,
        plan: list,
        results: dict,
        stream: bool,
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]]
    ) -> str:
        """Create final response from execution results."""
        import json
        
        # Build summary of execution
        execution_summary = []
        for step_num in sorted(results.keys()):
            result = results[step_num]
            step_info = result.get('step', {})
            if result.get('success'):
                execution_summary.append(f"Step {step_num} ({step_info.get('tool', 'unknown')}): Success")
            else:
                execution_summary.append(f"Step {step_num} ({step_info.get('tool', 'unknown')}): Failed - {result.get('error', 'Unknown error')}")
        
        synthesis_messages = [
            {"role": "system", "content": """You are a helpful assistant. Summarize the execution results into a clear, natural response.

Based on the plan execution results, provide a helpful response that:
1. Addresses what the user asked for
2. Summarizes what was accomplished
3. Highlights any important findings or results
4. Mentions any errors or issues that occurred

Be concise but informative."""},
            {"role": "user", "content": f"""User asked: {user_message}

Execution Summary:
{chr(10).join(execution_summary)}

Detailed Results:
{json.dumps(results, indent=2)[:2000]}  # Limit size

Provide a helpful, natural response based on what was accomplished."""}
        ]
        
        if stream and stream_callback:
            response = ""
            if self.verbose:
                print("\n💬 Final Response:")
                print("-" * 60)
            async for chunk in self.provider.chat_stream(
                messages=synthesis_messages,
                tools=None,
                model=self.model
            ):
                response += chunk
                if asyncio.iscoroutinefunction(stream_callback):
                    await stream_callback(chunk)
                else:
                    stream_callback(chunk)
            return response
        else:
            response = await self.provider.chat(messages=synthesis_messages, model=self.model, tools=None)
            if self.verbose:
                print("\n💬 Final Response:")
                print("-" * 60)
                print(response.content)
            return response.content
    
    async def _process_with_planning(
        self,
        user_message: str,
        session_key: str = "default",
        stream: bool = False,
        stream_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None,
    ) -> str:
        """Process message with planning: Plan → Execute → Synthesize."""
        import json
        session = self.sessions.get_or_create(session_key)
        
        if self.verbose:
            print("\n" + "="*60)
            print("🧠 PLANNING MODE ACTIVATED")
            print("="*60)
        
        # PHASE 1: Planning
        if self.verbose:
            print("\n📋 PHASE 1: Creating Execution Plan...")
            print("-" * 60)
        plan = await self._plan(user_message, session)
        
        if not plan:
            if self.verbose:
                print("⚠️  No plan generated. Falling back to standard mode.")
            # Temporarily disable planning to avoid recursion
            original_planning = self.use_planning
            self.use_planning = False
            result = await self.process_message(user_message, session_key, stream, stream_callback)
            self.use_planning = original_planning
            return result
        
        if self.verbose:
            print(f"\n✅ Plan created with {len(plan)} step(s)")
            print("\n📝 Execution Plan:")
            for step in plan:
                depends = f" (depends on: {step.get('depends_on', [])})" if step.get('depends_on') else ""
                print(f"   Step {step['step']}: {step.get('reasoning', 'No reasoning')}{depends}")
        
        # PHASE 2: Execution
        if self.verbose:
            print("\n" + "="*60)
            print("⚙️  PHASE 2: Executing Plan...")
            print("="*60)
        execution_results = await self._execute_plan(plan)
        
        # PHASE 3: Synthesis
        if self.verbose:
            print("\n" + "="*60)
            print("📊 PHASE 3: Synthesizing Results...")
            print("="*60)
        final_response = await self._synthesize(user_message, plan, execution_results, stream, stream_callback)
        
        # Save to session
        session.add_message("user", user_message)
        session.add_message("assistant", final_response)
        self.sessions.save(session)
        
        if self.verbose:
            print("\n" + "="*60)
            print("✅ COMPLETE")
            print("="*60 + "\n")
        
        return final_response