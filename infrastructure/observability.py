"""
Observability - LMNR (Laminar) integration for Deep Research Swarm.

Provides automatic tracing and observability for all agent operations
using the Laminar platform.

Usage:
    from infrastructure.observability import init_observability, observe

    # At application startup
    init_observability()

    # Decorate functions for tracing
    @observe()
    def my_function():
        ...
"""
import os
from functools import wraps
from typing import Any, Callable, Optional

from agno.utils.log import logger

from config import config


# Global flag to track initialization
_OBSERVABILITY_INITIALIZED = False
_LMNR_AVAILABLE = False

# Try to import LMNR
try:
    from lmnr import Laminar, observe as lmnr_observe
    _LMNR_AVAILABLE = True
except ImportError:
    Laminar = None
    lmnr_observe = None
    _LMNR_AVAILABLE = False

# Try to import OpenTelemetry Agno instrumentation
try:
    from opentelemetry.instrumentation.agno import AgnoInstrumentor
    _AGNO_INSTRUMENTOR_AVAILABLE = True
except ImportError:
    AgnoInstrumentor = None
    _AGNO_INSTRUMENTOR_AVAILABLE = False


def init_observability() -> bool:
    """
    Initialize LMNR observability if configured.
    
    Call this once at application startup to enable tracing.
    
    Returns:
        bool: True if observability was initialized, False otherwise
    """
    global _OBSERVABILITY_INITIALIZED
    
    if _OBSERVABILITY_INITIALIZED:
        logger.debug("[Observability] Already initialized")
        return True
    
    if not config.observability.lmnr_enabled:
        logger.info("[Observability] LMNR disabled (LMNR_PROJECT_API_KEY not set)")
        return False
    
    if not _LMNR_AVAILABLE:
        logger.warning("[Observability] LMNR package not installed - run: pip install lmnr")
        return False
    
    try:
        # Initialize Laminar
        api_key = config.observability.lmnr_project_api_key
        if not api_key:
            logger.warning("[Observability] LMNR_PROJECT_API_KEY not set")
            return False
        
        Laminar.initialize(project_api_key=api_key)
        logger.info("[Observability] LMNR initialized successfully")
        
        # Try to instrument Agno agents
        if _AGNO_INSTRUMENTOR_AVAILABLE:
            try:
                AgnoInstrumentor().instrument()
                logger.info("[Observability] Agno instrumentation enabled")
            except Exception as e:
                logger.warning(f"[Observability] Could not instrument Agno: {e}")
        else:
            logger.info("[Observability] OpenTelemetry Agno instrumentation not available")
        
        _OBSERVABILITY_INITIALIZED = True
        return True
        
    except Exception as e:
        logger.error(f"[Observability] Failed to initialize LMNR: {e}")
        return False


def observe(
    name: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Decorator to add observability tracing to a function.
    
    If LMNR is not available or not initialized, this is a no-op decorator.
    
    Args:
        name: Optional span name (defaults to function name)
        **kwargs: Additional arguments passed to lmnr.observe()
        
    Returns:
        Decorated function with tracing if available, otherwise original function
    """
    def decorator(func: Callable) -> Callable:
        # If LMNR is not available, return function as-is
        if not _LMNR_AVAILABLE or lmnr_observe is None:
            return func
        
        # If observability is not initialized and won't be, skip
        if not config.observability.lmnr_enabled:
            return func
        
        # Apply LMNR observe decorator
        span_name = name or func.__name__
        decorated = lmnr_observe(name=span_name, **kwargs)(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure observability is initialized
            if not _OBSERVABILITY_INITIALIZED:
                init_observability()
            return decorated(*args, **kwargs)
        
        return wrapper
    
    return decorator


def get_observability_status() -> dict:
    """
    Get the current status of observability configuration.
    
    Returns:
        dict: Status information about observability
    """
    return {
        "lmnr_enabled": config.observability.lmnr_enabled,
        "lmnr_available": _LMNR_AVAILABLE,
        "agno_instrumentor_available": _AGNO_INSTRUMENTOR_AVAILABLE,
        "initialized": _OBSERVABILITY_INITIALIZED,
        "api_key_set": bool(config.observability.lmnr_project_api_key),
    }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Observability Test ===\n")
    
    status = get_observability_status()
    
    print("Status:")
    for key, value in status.items():
        emoji = "✅" if value else "❌"
        print(f"  {emoji} {key}: {value}")
    
    print("\nAttempting initialization...")
    result = init_observability()
    
    if result:
        print("✅ Observability initialized successfully!")
        
        # Test observe decorator
        @observe(name="test_function")
        def test_function(x):
            return x * 2
        
        print(f"\nTesting decorated function: test_function(5) = {test_function(5)}")
    else:
        print("❌ Observability not initialized")
        print("\nTo enable:")
        print("  1. pip install lmnr")
        print("  2. Set LMNR_PROJECT_API_KEY in .env")
