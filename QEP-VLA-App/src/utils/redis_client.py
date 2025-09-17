#!/usr/bin/env python3
"""
Redis client utilities for QEP-VLA Platform
Caching, coordination, and pub/sub functionality
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from contextlib import asynccontextmanager

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global Redis client
_redis_client: Optional[redis.Redis] = None

async def init_redis() -> None:
    """Initialize Redis connection"""
    global _redis_client
    
    try:
        redis_url = settings.redis_url
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        await _redis_client.ping()
        logger.info("Redis connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        raise

async def close_redis() -> None:
    """Close Redis connection"""
    global _redis_client
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")

def get_redis_client() -> redis.Redis:
    """Get Redis client instance"""
    if not _redis_client:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client

# Cache operations
async def set_cache(
    key: str, 
    value: Any, 
    expire_seconds: Optional[int] = None
) -> bool:
    """Set cache value"""
    try:
        client = get_redis_client()
        serialized_value = json.dumps(value) if not isinstance(value, (str, int, float, bool)) else value
        return await client.set(key, serialized_value, ex=expire_seconds)
    except Exception as e:
        logger.error(f"Failed to set cache {key}: {e}")
        return False

async def get_cache(key: str) -> Optional[Any]:
    """Get cache value"""
    try:
        client = get_redis_client()
        value = await client.get(key)
        if value is None:
            return None
        
        # Try to deserialize JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
            
    except Exception as e:
        logger.error(f"Failed to get cache {key}: {e}")
        return None

async def delete_cache(key: str) -> bool:
    """Delete cache key"""
    try:
        client = get_redis_client()
        return bool(await client.delete(key))
    except Exception as e:
        logger.error(f"Failed to delete cache {key}: {e}")
        return False

async def clear_cache_pattern(pattern: str) -> int:
    """Clear cache keys matching pattern"""
    try:
        client = get_redis_client()
        keys = await client.keys(pattern)
        if keys:
            return await client.delete(*keys)
        return 0
    except Exception as e:
        logger.error(f"Failed to clear cache pattern {pattern}: {e}")
        return 0

# Hash operations
async def set_hash_cache(
    name: str, 
    mapping: Dict[str, Any], 
    expire_seconds: Optional[int] = None
) -> bool:
    """Set hash cache"""
    try:
        client = get_redis_client()
        # Serialize non-primitive values
        serialized_mapping = {}
        for k, v in mapping.items():
            if not isinstance(v, (str, int, float, bool)):
                serialized_mapping[k] = json.dumps(v)
            else:
                serialized_mapping[k] = v
        
        result = await client.hset(name, mapping=serialized_mapping)
        if expire_seconds:
            await client.expire(name, expire_seconds)
        return bool(result)
    except Exception as e:
        logger.error(f"Failed to set hash cache {name}: {e}")
        return False

async def get_hash_cache(name: str, key: Optional[str] = None) -> Optional[Any]:
    """Get hash cache value"""
    try:
        client = get_redis_client()
        if key:
            value = await client.hget(name, key)
        else:
            value = await client.hgetall(name)
        
        if value is None:
            return None
        
        # Try to deserialize JSON values
        if isinstance(value, dict):
            deserialized = {}
            for k, v in value.items():
                try:
                    deserialized[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    deserialized[k] = v
            return deserialized
        else:
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
    except Exception as e:
        logger.error(f"Failed to get hash cache {name}: {e}")
        return None

async def get_all_hash_cache(name: str) -> Dict[str, Any]:
    """Get all hash cache values"""
    try:
        client = get_redis_client()
        values = await client.hgetall(name)
        
        # Deserialize JSON values
        deserialized = {}
        for k, v in values.items():
            try:
                deserialized[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                deserialized[k] = v
        
        return deserialized
        
    except Exception as e:
        logger.error(f"Failed to get all hash cache {name}: {e}")
        return {}

# Counter operations
async def increment_counter(key: str, amount: int = 1, expire_seconds: Optional[int] = None) -> int:
    """Increment counter"""
    try:
        client = get_redis_client()
        result = await client.incr(key, amount)
        if expire_seconds:
            await client.expire(key, expire_seconds)
        return result
    except Exception as e:
        logger.error(f"Failed to increment counter {key}: {e}")
        return 0

async def get_counter(key: str) -> int:
    """Get counter value"""
    try:
        client = get_redis_client()
        value = await client.get(key)
        return int(value) if value else 0
    except Exception as e:
        logger.error(f"Failed to get counter {key}: {e}")
        return 0

# List operations
async def set_list_cache(
    key: str, 
    values: List[Any], 
    expire_seconds: Optional[int] = None
) -> bool:
    """Set list cache"""
    try:
        client = get_redis_client()
        # Clear existing list
        await client.delete(key)
        
        # Serialize non-primitive values
        serialized_values = []
        for v in values:
            if not isinstance(v, (str, int, float, bool)):
                serialized_values.append(json.dumps(v))
            else:
                serialized_values.append(v)
        
        if serialized_values:
            await client.rpush(key, *serialized_values)
        
        if expire_seconds:
            await client.expire(key, expire_seconds)
        
        return True
    except Exception as e:
        logger.error(f"Failed to set list cache {key}: {e}")
        return False

async def get_list_cache(key: str) -> List[Any]:
    """Get list cache values"""
    try:
        client = get_redis_client()
        values = await client.lrange(key, 0, -1)
        
        # Deserialize JSON values
        deserialized = []
        for v in values:
            try:
                deserialized.append(json.loads(v))
            except (json.JSONDecodeError, TypeError):
                deserialized.append(v)
        
        return deserialized
        
    except Exception as e:
        logger.error(f"Failed to get list cache {key}: {e}")
        return []

async def add_to_list_cache(
    key: str, 
    value: Any, 
    expire_seconds: Optional[int] = None
) -> bool:
    """Add value to list cache"""
    try:
        client = get_redis_client()
        
        # Serialize non-primitive values
        if not isinstance(value, (str, int, float, bool)):
            serialized_value = json.dumps(value)
        else:
            serialized_value = value
        
        await client.rpush(key, serialized_value)
        
        if expire_seconds:
            await client.expire(key, expire_seconds)
        
        return True
    except Exception as e:
        logger.error(f"Failed to add to list cache {key}: {e}")
        return False

# Pub/Sub operations
async def publish_message(channel: str, message: Any) -> int:
    """Publish message to channel"""
    try:
        client = get_redis_client()
        serialized_message = json.dumps(message) if not isinstance(message, (str, int, float, bool)) else message
        return await client.publish(channel, serialized_message)
    except Exception as e:
        logger.error(f"Failed to publish message to {channel}: {e}")
        return 0

async def subscribe_to_channel(channel: str, callback) -> None:
    """Subscribe to channel with callback"""
    try:
        client = get_redis_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                except (json.JSONDecodeError, TypeError):
                    data = message['data']
                
                await callback(channel, data)
                
    except Exception as e:
        logger.error(f"Failed to subscribe to {channel}: {e}")

# Health check
async def health_check() -> Dict[str, Any]:
    """Redis health check"""
    try:
        if not _redis_client:
            return {"status": "unhealthy", "error": "Redis not initialized"}
        
        # Test connection
        await _redis_client.ping()
        
        # Get info
        info = await _redis_client.info()
        
        return {
            "status": "healthy",
            "connected_clients": info.get('connected_clients', 0),
            "used_memory_human": info.get('used_memory_human', '0B'),
            "uptime_in_seconds": info.get('uptime_in_seconds', 0)
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Cache statistics
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    try:
        client = get_redis_client()
        info = await client.info()
        
        return {
            "total_commands_processed": info.get('total_commands_processed', 0),
            "total_connections_received": info.get('total_connections_received', 0),
            "total_net_input_bytes": info.get('total_net_input_bytes', 0),
            "total_net_output_bytes": info.get('total_net_output_bytes', 0),
            "keyspace_hits": info.get('keyspace_hits', 0),
            "keyspace_misses": info.get('keyspace_misses', 0),
            "hit_rate": info.get('keyspace_hits', 0) / max(info.get('keyspace_misses', 1), 1)
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {}

# Utility functions
async def cache_exists(key: str) -> bool:
    """Check if cache key exists"""
    try:
        client = get_redis_client()
        return bool(await client.exists(key))
    except Exception as e:
        logger.error(f"Failed to check cache existence {key}: {e}")
        return False

async def get_cache_ttl(key: str) -> int:
    """Get cache TTL in seconds"""
    try:
        client = get_redis_client()
        ttl = await client.ttl(key)
        return ttl if ttl > 0 else -1
    except Exception as e:
        logger.error(f"Failed to get cache TTL {key}: {e}")
        return -1

async def set_cache_ttl(key: str, seconds: int) -> bool:
    """Set cache TTL"""
    try:
        client = get_redis_client()
        return bool(await client.expire(key, seconds))
    except Exception as e:
        logger.error(f"Failed to set cache TTL {key}: {e}")
        return False
