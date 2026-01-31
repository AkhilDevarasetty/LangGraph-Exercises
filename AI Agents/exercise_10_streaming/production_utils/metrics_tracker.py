"""
Streaming Metrics Tracker

Tracks performance metrics for streaming LLM responses including
TTFT (Time To First Token), latency, throughput, and error rates.
"""

import time
from typing import List, Dict, Optional
from colorama import Fore, Style


class StreamingMetrics:
    """
    Track and analyze streaming performance metrics.

    Collects metrics for each streaming session including:
    - Time to first token (TTFT)
    - Total latency
    - Token count
    - Tokens per second
    - Error rate

    Example:
        metrics = StreamingMetrics()

        session = metrics.start_session()
        # ... streaming happens ...
        metrics.record_first_token(session)
        session['token_count'] = 150
        metrics.finish_session(session)

        # Display summary every 5 requests
        if len(metrics.sessions) % 5 == 0:
            metrics.display_summary()
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.sessions: List[Dict] = []

    def start_session(self) -> Dict:
        """
        Start tracking a new streaming session.

        Returns:
            Session dictionary to be updated during streaming
        """
        return {
            "start_time": time.time(),
            "first_token_time": None,
            "end_time": None,
            "token_count": 0,
            "error": None,
        }

    def record_first_token(self, session: Dict) -> None:
        """
        Record when the first token arrives.

        Args:
            session: Session dictionary from start_session()
        """
        if session["first_token_time"] is None:
            session["first_token_time"] = time.time()

    def finish_session(self, session: Dict) -> None:
        """
        Complete a session and add it to history.

        Args:
            session: Session dictionary to finalize
        """
        session["end_time"] = time.time()
        self.sessions.append(session)

    def get_summary(self) -> Dict:
        """
        Calculate aggregate metrics across all sessions.

        Returns:
            Dictionary with summary statistics
        """
        if not self.sessions:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_ttft": 0,
                "avg_latency": 0,
                "avg_tokens": 0,
                "avg_tokens_per_sec": 0,
                "error_rate": 0,
            }

        successful = [
            s
            for s in self.sessions
            if s["error"] is None and s["first_token_time"] is not None
        ]
        failed = [s for s in self.sessions if s["error"] is not None]

        if not successful:
            return {
                "total_requests": len(self.sessions),
                "successful_requests": 0,
                "failed_requests": len(failed),
                "avg_ttft": 0,
                "avg_latency": 0,
                "avg_tokens": 0,
                "avg_tokens_per_sec": 0,
                "error_rate": 100.0,
            }

        # Calculate averages
        avg_ttft = sum(
            s["first_token_time"] - s["start_time"] for s in successful
        ) / len(successful)
        avg_latency = sum(s["end_time"] - s["start_time"] for s in successful) / len(
            successful
        )
        avg_tokens = sum(s["token_count"] for s in successful) / len(successful)

        # Calculate tokens per second
        tokens_per_sec = []
        for s in successful:
            duration = s["end_time"] - s["first_token_time"]
            if duration > 0:
                tokens_per_sec.append(s["token_count"] / duration)
        avg_tokens_per_sec = (
            sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0
        )

        error_rate = len(failed) / len(self.sessions) * 100

        return {
            "total_requests": len(self.sessions),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "avg_ttft": avg_ttft,
            "avg_latency": avg_latency,
            "avg_tokens": avg_tokens,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "error_rate": error_rate,
        }

    def display_summary(self) -> None:
        """Display a formatted metrics dashboard."""
        summary = self.get_summary()

        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"📊 STREAMING METRICS DASHBOARD")
        print(f"{'=' * 60}")
        print(f"Total Requests:      {summary['total_requests']}")
        print(
            f"Successful:          {summary['successful_requests']} ({100 - summary['error_rate']:.1f}%)"
        )
        print(
            f"Failed:              {summary['failed_requests']} ({summary['error_rate']:.1f}%)"
        )
        print(f"─" * 60)
        print(f"Avg TTFT:            {summary['avg_ttft']:.2f}s")
        print(f"Avg Latency:         {summary['avg_latency']:.2f}s")
        print(f"Avg Tokens:          {summary['avg_tokens']:.0f}")
        print(f"Avg Tokens/sec:      {summary['avg_tokens_per_sec']:.1f}")
        print(f"{'=' * 60}{Style.RESET_ALL}\n")

    def reset(self) -> None:
        """Clear all session history."""
        self.sessions = []
