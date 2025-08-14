import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Optional
import logging


class RateLimiter:
    """Thread-safe rate limiter with sliding window implementation"""

    def __init__(self, max_calls: int, period: int):  # Fixed __init__
        self.max_calls = max_calls
        self.period = period
        self.calls: deque = deque()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a rate limit slot. Returns True if allowed, False if rate limited."""
        async with self.lock:
            now = time.time()
            # Clean up old calls outside the time window
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()

            # Check if we can make another call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

    async def wait_if_needed(self) -> Optional[float]:
        """Wait until a slot is available. Returns wait time in seconds."""
        async with self.lock:
            now = time.time()
            # Clean up old calls
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()

            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return None

            # Calculate wait time until oldest call expires
            wait_time = (self.calls[0] + self.period) - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return wait_time
            return None

    def get_remaining_calls(self) -> int:
        """Get number of remaining calls in current period"""
        now = time.time()
        # Remove expired calls
        while self.calls and self.calls[0] <= now - self.period:
            self.calls.popleft()
        return max(0, self.max_calls - len(self.calls))

    def get_reset_time(self) -> Optional[float]:
        """Get time when next slot will be available"""
        if not self.calls or len(self.calls) < self.max_calls:
            return None
        return self.calls[0] + self.period
