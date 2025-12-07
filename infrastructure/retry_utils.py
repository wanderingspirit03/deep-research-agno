"""
Retry utilities with exponential backoff for timeout handling.

Provides decorators and utilities for handling transient failures
in LLM API calls and long-running research operations.
"""
import time
import asyncio
import functools
from typing import Callable, TypeVar, Tuple, Optional, Union
from agno.utils.log import logger

T = TypeVar('T')


# =============================================================================
# Synchronous Retry Decorator
# =============================================================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt) called on each retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @with_retry(max_retries=3, base_delay=5.0)
        def call_api():
            return api.request()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(e, attempt)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed. Last error: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Async Retry Decorator
# =============================================================================

def with_async_retry(
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Async decorator for retrying operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt) called on each retry
        
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        
                        logger.warning(
                            f"Async attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        if on_retry:
                            on_retry(e, attempt)
                        
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} async attempts failed. Last error: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Retry Context Manager
# =============================================================================

class RetryContext:
    """
    Context manager for manual retry logic.
    
    Example:
        async with RetryContext(max_retries=3) as ctx:
            while ctx.should_retry():
                try:
                    result = await api_call()
                    break
                except Exception as e:
                    await ctx.handle_error(e)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 5.0,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
    
    def should_retry(self) -> bool:
        """Check if another retry attempt should be made"""
        return self.attempt < self.max_retries
    
    def handle_error(self, error: Exception) -> float:
        """
        Handle an error and return the delay before next retry.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Delay in seconds before next retry
            
        Raises:
            The error if max retries exceeded
        """
        self.last_exception = error
        self.attempt += 1
        
        if self.attempt >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded: {error}")
            raise error
        
        delay = min(self.base_delay * (2 ** (self.attempt - 1)), self.max_delay)
        logger.warning(
            f"Attempt {self.attempt}/{self.max_retries} failed: {error}. "
            f"Will retry in {delay:.1f}s..."
        )
        
        return delay
    
    async def handle_error_async(self, error: Exception):
        """Handle error and sleep asynchronously"""
        delay = self.handle_error(error)
        await asyncio.sleep(delay)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """
    Calculate delay for a retry attempt with exponential backoff.
    
    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Add random jitter to prevent thundering herd
        
    Returns:
        Delay in seconds
    """
    import random
    
    delay = min(base_delay * (2 ** attempt), max_delay)
    
    if jitter:
        # Add ±25% jitter
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


def is_retriable_error(error: Exception) -> bool:
    """
    Check if an error is retriable (transient).
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is likely transient and worth retrying
    """
    # Common transient error patterns
    transient_patterns = [
        "timeout",
        "rate limit",
        "429",  # Too Many Requests
        "500",  # Internal Server Error
        "502",  # Bad Gateway
        "503",  # Service Unavailable
        "504",  # Gateway Timeout
        "connection",
        "network",
        "temporarily unavailable",
    ]
    
    error_str = str(error).lower()
    
    return any(pattern in error_str for pattern in transient_patterns)


# =============================================================================
# LLM-Specific Retry Decorators
# =============================================================================

def with_llm_retry(
    max_retries: int = 3,
    base_delay: float = 5.0,
):
    """
    Specialized retry decorator for LLM API calls.
    
    Handles common LLM API errors like rate limits and timeouts.
    """
    # Import common LLM exceptions
    try:
        import openai
        llm_exceptions = (
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
    except ImportError:
        llm_exceptions = ()
    
    # Add general exceptions
    all_exceptions = (*llm_exceptions, TimeoutError, ConnectionError)
    
    return with_retry(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=120.0,  # Up to 2 minutes for LLM rate limits
        exceptions=all_exceptions if all_exceptions else (Exception,),
    )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("=== Retry Utils Test ===\n")
    
    # Test sync retry
    call_count = 0
    
    @with_retry(max_retries=3, base_delay=1.0)
    def flaky_function():
        global call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Simulated failure")
        return "Success!"
    
    print("Testing sync retry...")
    try:
        result = flaky_function()
        print(f"✅ Result: {result} (after {call_count} attempts)")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test backoff calculation
    print("\nBackoff delays:")
    for i in range(5):
        delay = calculate_backoff_delay(i, base_delay=2.0, jitter=False)
        print(f"  Attempt {i}: {delay:.1f}s")
    
    print("\n✅ Retry utils tests complete!")



