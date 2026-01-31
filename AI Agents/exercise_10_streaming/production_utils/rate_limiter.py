"""
Rate Limiter for API Request Throttling

Implements a token bucket algorithm to prevent API overload.
"""

import time
import asyncio
from typing import List


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request frequency.

    Prevents exceeding API rate limits by tracking request timestamps
    and enforcing a maximum number of requests per time window.

    Example:
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        async def make_request():
            await limiter.acquire()  # Wait if rate limit exceeded
            # Make API call...
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            window_seconds: Time window in seconds (default: 60s = 1 minute)
        """
        self.requests: List[float] = []  # Timestamps of recent requests
        self.max_requests = max_requests
        self.window = window_seconds

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.

        If the rate limit is exceeded, this method will sleep until
        a request slot becomes available.
        """
        now = time.time()

        # Remove requests outside the current time window
        self.requests = [r for r in self.requests if now - r < self.window]

        # Check if we've hit the rate limit
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait
            oldest_request = self.requests[0]
            wait_time = self.window - (now - oldest_request)

            print(
                f"⏳ Rate limit reached ({self.max_requests} requests/{self.window}s). Waiting {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)

            # Recursively try again after waiting
            await self.acquire()
            return

        # Record this request
        self.requests.append(now)

    def get_current_usage(self) -> int:
        """
        Get the current number of requests in the time window.

        Returns:
            Number of active requests in the current window
        """
        now = time.time()
        self.requests = [r for r in self.requests if now - r < self.window]
        return len(self.requests)

    def reset(self) -> None:
        """Clear all request history."""
        self.requests = []
