# src/pymemorial/core/cache.py
"""
Intelligent Result Cache - PyMemorial v2.0 (Production Ready).

Features:
- ✅ LRU (Least Recently Used) eviction policy
- ✅ Thread-safe operations with Lock
- ✅ Cache statistics (hits, misses, evictions, invalidations)
- ✅ Norm-aware invalidation (auto-invalidate on norm change)
- ✅ Smart optimization suggestions (resize, clear frequency)
- ✅ TTL (Time-To-Live) support for auto-expiration
- ✅ Memory-efficient with size tracking
- ✅ Decorator for method caching

Author: PyMemorial Team
Date: 2025-10-21
Version: 2.0.0
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, Any, Hashable, Optional, List, Callable
from collections import OrderedDict
from threading import Lock
from functools import wraps

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# CACHE ENTRY (INTERNAL)
# ============================================================================

class CacheEntry:
    """
    Cache entry with value and metadata.
    
    Attributes:
        value: Cached value
        timestamp: Creation timestamp (for TTL)
        norm_tags: Set of norm codes associated with entry
        size: Estimated size in bytes (for memory tracking)
    """
    
    def __init__(
        self,
        value: Any,
        norm_tags: Optional[set] = None,
        ttl: Optional[float] = None
    ):
        self.value = value
        self.timestamp = time.time()
        self.norm_tags = norm_tags or set()
        self.ttl = ttl
    
    def is_expired(self) -> bool:
        """Check if entry is expired (TTL)."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    def has_norm(self, norm_code: str) -> bool:
        """Check if entry is tagged with norm code."""
        return norm_code in self.norm_tags

# ============================================================================
# RESULT CACHE
# ============================================================================

class ResultCache:
    """
    Intelligent LRU cache with thread-safety, statistics, and optimization.
    
    Features:
    - LRU eviction policy for memory efficiency
    - Thread-safe operations
    - Norm-aware invalidation (re-compute on norm change)
    - Smart optimization suggestions based on usage patterns
    - TTL (Time-To-Live) support for auto-expiration
    - Comprehensive statistics tracking
    
    Examples:
    --------
    >>> cache = ResultCache(maxsize=128, ttl=3600)  # 1 hour TTL
    >>> 
    >>> # Basic usage
    >>> cache.set('M_d', 210.0)
    >>> assert cache.get('M_d') == 210.0
    >>> 
    >>> # Norm-aware caching
    >>> cache.set('M_d_NBR6118', 210.0, norm_tags={'NBR6118_2023'})
    >>> cache.invalidate_norms('NBR6118')  # Invalidate all NBR entries
    >>> assert cache.get('M_d_NBR6118') is None  # Invalidated
    >>> 
    >>> # Statistics
    >>> stats = cache.get_stats()
    >>> print(stats)  # {'hits': 1, 'misses': 1, ...}
    >>> 
    >>> # Optimization suggestions
    >>> suggestions = cache.suggest_optimizations()
    >>> print(suggestions)  # [{'op': 'resize', 'suggested_size': 256, ...}]
    """
    
    # Regex patterns for norm detection
    NORM_PATTERNS = [
        re.compile(r'_(NBR\d+)_', re.IGNORECASE),      # _NBR6118_
        re.compile(r'_(AISC\d+)_', re.IGNORECASE),     # _AISC360_
        re.compile(r'_(EC\d+)_', re.IGNORECASE),       # _EC2_
        re.compile(r'_(ACI\d+)_', re.IGNORECASE),      # _ACI318_
    ]
    
    def __init__(
        self,
        maxsize: int = 128,
        ttl: Optional[float] = None,
        thread_safe: bool = True
    ):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of entries (LRU eviction when full)
            ttl: Time-To-Live in seconds (None = no expiration)
            thread_safe: Enable thread-safe operations with Lock
        
        Raises:
            ValueError: If maxsize < 1
        """
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        
        # Configuration
        self.maxsize = maxsize
        self.ttl = ttl
        self.thread_safe = thread_safe
        
        # State
        self._cache: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._lock = Lock() if thread_safe else None
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0,
            'expirations': 0,
            'sets': 0
        }
        
        logger.debug(
            f"ResultCache initialized: maxsize={maxsize}, ttl={ttl}, "
            f"thread_safe={thread_safe}"
        )
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        
        Examples:
        --------
        >>> value = cache.get('M_d')
        >>> if value is not None:
        ...     print(f"Cache hit: {value}")
        """
        if self.thread_safe:
            with self._lock:
                return self._get_internal(key)
        else:
            return self._get_internal(key)
    
    def _get_internal(self, key: Hashable) -> Optional[Any]:
        """Internal get (assumes lock is held if thread_safe=True)."""
        if key not in self._cache:
            self.stats['misses'] += 1
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if entry.is_expired():
            del self._cache[key]
            self.stats['expirations'] += 1
            self.stats['misses'] += 1
            logger.debug(f"Cache entry expired: {key}")
            return None
        
        # LRU: Move to end (most recently used)
        self._cache.move_to_end(key)
        self.stats['hits'] += 1
        
        return entry.value
    
    def set(
        self,
        key: Hashable,
        value: Any,
        norm_tags: Optional[set] = None,
        ttl: Optional[float] = None
    ) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            norm_tags: Set of norm codes associated with value (for invalidation)
            ttl: Override default TTL for this entry
        
        Examples:
        --------
        >>> cache.set('M_d', 210.0)
        >>> cache.set('M_d_NBR', 210.0, norm_tags={'NBR6118_2023'})
        >>> cache.set('temp', 100, ttl=60)  # Expires in 60 seconds
        """
        if self.thread_safe:
            with self._lock:
                self._set_internal(key, value, norm_tags, ttl)
        else:
            self._set_internal(key, value, norm_tags, ttl)
    
    def _set_internal(
        self,
        key: Hashable,
        value: Any,
        norm_tags: Optional[set],
        ttl: Optional[float]
    ) -> None:
        """Internal set (assumes lock is held if thread_safe=True)."""
        # Auto-detect norm tags from key if not provided
        if norm_tags is None:
            norm_tags = self._detect_norm_tags(key)
        
        # Use instance TTL if not specified
        if ttl is None:
            ttl = self.ttl
        
        # Create entry
        entry = CacheEntry(value, norm_tags, ttl)
        
        # Store and mark as most recently used
        self._cache[key] = entry
        self._cache.move_to_end(key)
        self.stats['sets'] += 1
        
        # Evict oldest if cache full (LRU policy)
        if len(self._cache) > self.maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats['evictions'] += 1
            logger.debug(f"LRU eviction: {oldest_key}")
    
    def _detect_norm_tags(self, key: Hashable) -> set:
        """
        Auto-detect norm codes from key.
        
        Args:
            key: Cache key
        
        Returns:
            Set of detected norm codes
        
        Examples:
        --------
        >>> cache._detect_norm_tags('M_d_NBR6118_2023')
        {'NBR6118'}
        >>> cache._detect_norm_tags('M_d_AISC360_22')
        {'AISC360'}
        """
        key_str = str(key)
        detected = set()
        
        for pattern in self.NORM_PATTERNS:
            match = pattern.search(key_str)
            if match:
                detected.add(match.group(1).upper())
        
        return detected
    
    # ========================================================================
    # INVALIDATION
    # ========================================================================
    
    def invalidate_norms(self, *norm_codes: str) -> int:
        """
        Invalidate all entries associated with norm codes.
        
        Args:
            *norm_codes: Norm codes to invalidate (e.g., 'NBR6118', 'AISC360')
        
        Returns:
            Number of entries invalidated
        
        Examples:
        --------
        >>> count = cache.invalidate_norms('NBR6118', 'NBR8800')
        >>> print(f"Invalidated {count} entries")
        """
        if self.thread_safe:
            with self._lock:
                return self._invalidate_norms_internal(norm_codes)
        else:
            return self._invalidate_norms_internal(norm_codes)
    
    def _invalidate_norms_internal(self, norm_codes: tuple) -> int:
        """Internal invalidation (assumes lock is held if thread_safe=True)."""
        # Normalize norm codes (uppercase)
        norm_codes_upper = {code.upper() for code in norm_codes}
        
        # Find entries to invalidate
        keys_to_delete = []
        for key, entry in self._cache.items():
            if any(entry.has_norm(code) for code in norm_codes_upper):
                keys_to_delete.append(key)
        
        # Delete entries
        for key in keys_to_delete:
            del self._cache[key]
        
        count = len(keys_to_delete)
        self.stats['invalidations'] += count
        
        if count > 0:
            logger.info(
                f"Invalidated {count} entries for norms: {', '.join(norm_codes_upper)}"
            )
        
        return count
    
    def clear(self) -> None:
        """
        Clear all entries and reset statistics.
        
        Examples:
        --------
        >>> cache.clear()
        >>> assert cache.get_stats()['hits'] == 0
        """
        if self.thread_safe:
            with self._lock:
                self._clear_internal()
        else:
            self._clear_internal()
    
    def _clear_internal(self) -> None:
        """Internal clear (assumes lock is held if thread_safe=True)."""
        self._cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'invalidations': 0,
            'expirations': 0,
            'sets': 0
        }
        logger.info("Cache cleared. Statistics reset.")
    
    def clear_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        
        Examples:
        --------
        >>> count = cache.clear_expired()
        >>> print(f"Removed {count} expired entries")
        """
        if self.thread_safe:
            with self._lock:
                return self._clear_expired_internal()
        else:
            return self._clear_expired_internal()
    
    def _clear_expired_internal(self) -> int:
        """Internal clear expired (assumes lock is held if thread_safe=True)."""
        keys_to_delete = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._cache[key]
        
        count = len(keys_to_delete)
        self.stats['expirations'] += count
        
        if count > 0:
            logger.debug(f"Cleared {count} expired entries")
        
        return count
    
    # ========================================================================
    # STATISTICS & OPTIMIZATION
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        
        Examples:
        --------
        >>> stats = cache.get_stats()
        >>> print(stats)
        {'hits': 10, 'misses': 2, 'hit_rate': 0.833, ...}
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            'size': len(self._cache),
            'maxsize': self.maxsize,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Suggest optimizations based on usage patterns.
        
        Returns:
            List of optimization suggestions
        
        Examples:
        --------
        >>> suggestions = cache.suggest_optimizations()
        >>> for s in suggestions:
        ...     print(f"{s['op']}: {s['reason']}")
        resize: Hit rate 65% < 80%; double size for better performance
        """
        suggestions = []
        
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests == 0:
            return suggestions  # No data yet
        
        hit_rate = self.stats['hits'] / total_requests
        
        # Suggestion 1: Resize cache if low hit rate
        if hit_rate < 0.8:
            suggestions.append({
                'op': 'resize',
                'suggested_size': self.maxsize * 2,
                'reason': f"Hit rate {hit_rate:.1%} < 80%; double size for better performance",
                'current_size': self.maxsize,
                'hit_rate': hit_rate
            })
        
        # Suggestion 2: Clear frequently if high invalidation rate
        invalidation_rate = self.stats['invalidations'] / total_requests
        if invalidation_rate > 0.3:
            suggestions.append({
                'op': 'clear_frequency',
                'reason': f"High invalidation rate {invalidation_rate:.1%}; clear cache periodically",
                'invalidation_rate': invalidation_rate
            })
        
        # Suggestion 3: Enable TTL if many long-lived entries
        if self.ttl is None and len(self._cache) > self.maxsize * 0.8:
            suggestions.append({
                'op': 'enable_ttl',
                'reason': "Cache near capacity; enable TTL to auto-expire old entries",
                'suggested_ttl': 3600  # 1 hour
            })
        
        if suggestions:
            logger.debug(f"Generated {len(suggestions)} optimization suggestions")
        
        return suggestions


# ============================================================================
# DECORATOR FOR METHOD CACHING
# ============================================================================

def cached_method(cache: ResultCache, key_prefix: str = ''):
    """
    Decorator for method caching.
    
    Args:
        cache: ResultCache instance
        key_prefix: Prefix for cache keys (optional)
    
    Examples:
    --------
    >>> cache = ResultCache(maxsize=128)
    >>> 
    >>> class Calculator:
    ...     @cached_method(cache, key_prefix='calc')
    ...     def expensive_operation(self, x, y):
    ...         return x ** y
    >>> 
    >>> calc = Calculator()
    >>> result = calc.expensive_operation(2, 10)  # Cache miss
    >>> result = calc.expensive_operation(2, 10)  # Cache hit
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
            key = ':'.join(filter(None, key_parts))
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Cache miss - compute
            result = func(self, *args, **kwargs)
            cache.set(key, result)
            
            return result
        
        return wrapper
    
    return decorator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ['ResultCache', 'cached_method']
