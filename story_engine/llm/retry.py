"""Retry policies for LLM generation attempts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Policy for adjusting parameters across LLM generation attempts.

    Attributes:
        max_attempts: Maximum number of generation attempts.
        base_temperature: Starting temperature for first attempt.
    """
    max_attempts: int = 2
    base_temperature: float = 0.8

    def temperature_for(self, attempt: int) -> float:
        """Get adjusted temperature for a specific attempt.

        Decreases temperature with each retry to encourage consistency.

        Args:
            attempt: Attempt number (0-indexed).

        Returns:
            Adjusted temperature value.
        """
        return max(0.1, self.base_temperature - (attempt * 0.25))
