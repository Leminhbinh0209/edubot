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