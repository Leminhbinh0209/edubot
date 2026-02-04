import asyncio
import re
import time
from typing import Callable

class StreamSmoother:
    def __init__(
        self,
        callback: Callable[[str], None],
        min_chunk_chars: int = 15,
        max_chunk_chars: int = 80,
        base_delay: float = 0.03,
        char_delay: float = 0.008,  # delay per character
    ):
        self.callback = callback
        self.buffer = ""
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.base_delay = base_delay
        self.char_delay = char_delay
        self.boundary_re = re.compile(r'[.!?]\s+|[,;:]\s+|\n+')
        self.first_flush = True
    
    async def push(self, text: str):
        self.buffer += text
        
        # Always flush on max chars
        if len(self.buffer) >= self.max_chunk_chars:
            await self._flush_at_boundary()
            return
        
        # Flush at natural boundaries if we have enough content
        if len(self.buffer) >= self.min_chunk_chars:
            match = self.boundary_re.search(self.buffer)
            if match:
                await self._flush_at_boundary(match.end())
    
    async def _flush_at_boundary(self, pos: int = None):
        if not self.buffer:
            return
        
        # Determine what to flush
        if pos is None:
            to_flush = self.buffer
            self.buffer = ""
        else:
            to_flush = self.buffer[:pos]
            self.buffer = self.buffer[pos:]
        
        # First chunk is instant, then add natural delays
        if not self.first_flush:
            # Delay based on chunk length for natural feel
            delay = self.base_delay + (len(to_flush) * self.char_delay)
            await asyncio.sleep(min(delay, 0.15))  # cap at 150ms
        
        self.callback(to_flush)
        self.first_flush = False
    
    async def flush_final(self):
        """Flush any remaining buffer"""
        if self.buffer:
            if not self.first_flush:
                await asyncio.sleep(self.base_delay)
            self.callback(self.buffer)
            self.buffer = ""
