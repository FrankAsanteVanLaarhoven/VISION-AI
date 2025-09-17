#!/usr/bin/env python3
"""
Database utilities for QEP-VLA Platform
PostgreSQL connection management and table operations
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import asyncpg
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

def _parse_postgres_url(url: str) -> Dict[str, str]:
    """Parse PostgreSQL URL into components"""
    parsed = urlparse(url)
    return {
        'host': parsed.hostname or 'localhost',
        'port': parsed.port or 5432,
        'user': parsed.username or 'qepvla',
        'password': parsed.password or 'qepvla123',
        'database': parsed.path.lstrip('/') or 'qepvla'
    }

async def init_database() -> None:
    """Initialize database connection pool"""
    global _pool
    
    try:
        # Parse PostgreSQL URL
        db_config = _parse_postgres_url(settings.postgres_url)
        
        # Create connection pool
        _pool = await asyncpg.create_pool(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            min_size=5,
            max_size=20,
            command_timeout=60,
            server_settings={
                'application_name': 'qep_vla_platform'
            }
        )
        
        logger.info("Database connection pool initialized successfully")
        
        # Test connection
        async with _pool.acquire() as conn:
            await conn.execute('SELECT 1')
            logger.info("Database connection test successful")
        
        # Initialize tables if they don't exist
        await _init_tables()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

async def _init_tables() -> None:
    """Initialize database tables"""
    try:
        async with get_connection() as conn:
            # Create tables if they don't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    operation_type VARCHAR(50) NOT NULL,
                    processing_time_ms FLOAT,
                    confidence_score FLOAT,
                    privacy_level VARCHAR(20),
                    quantum_enhanced BOOLEAN,
                    model_complexity VARCHAR(20),
                    safety_fallback BOOLEAN,
                    metadata JSONB
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS privacy_transforms (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    transform_type VARCHAR(50) NOT NULL,
                    privacy_budget_epsilon FLOAT,
                    privacy_budget_delta FLOAT,
                    processing_time_ms FLOAT,
                    input_size INTEGER,
                    output_size INTEGER,
                    quantum_enhancement_factor FLOAT,
                    metadata JSONB
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS federated_training (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    round_number INTEGER NOT NULL,
                    participating_agents INTEGER,
                    validated_agents INTEGER,
                    validation_accuracy FLOAT,
                    validation_loss FLOAT,
                    privacy_budget_epsilon FLOAT,
                    processing_time_sec FLOAT,
                    aggregation_method VARCHAR(50),
                    blockchain_validation BOOLEAN,
                    metadata JSONB
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    component VARCHAR(50) NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT,
                    metric_unit VARCHAR(20),
                    metadata JSONB
                )
            """)
            
            # Create indexes for better performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inference_logs_timestamp 
                ON inference_logs(timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_privacy_transforms_timestamp 
                ON privacy_transforms(timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_federated_training_round 
                ON federated_training(round_number)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_metrics_component 
                ON system_metrics(component, timestamp)
            """)
            
            logger.info("Database tables initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize tables: {e}")
        raise

@asynccontextmanager
async def get_connection():
    """Get database connection from pool"""
    if not _pool:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    conn = await _pool.acquire()
    try:
        yield conn
    finally:
        await _pool.release(conn)

async def close_database() -> None:
    """Close database connection pool"""
    global _pool
    
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")

async def log_inference(
    operation_type: str,
    processing_time_ms: float,
    confidence_score: Optional[float] = None,
    privacy_level: Optional[str] = None,
    quantum_enhanced: bool = False,
    model_complexity: Optional[str] = None,
    safety_fallback: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log inference operation"""
    try:
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO inference_logs (
                    operation_type, processing_time_ms, confidence_score,
                    privacy_level, quantum_enhanced, model_complexity,
                    safety_fallback, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, operation_type, processing_time_ms, confidence_score,
                 privacy_level, quantum_enhanced, model_complexity,
                 safety_fallback, json.dumps(metadata) if metadata else None)
    except Exception as e:
        logger.error(f"Failed to log inference: {e}")

async def log_privacy_transform(
    transform_type: str,
    privacy_budget_epsilon: float,
    privacy_budget_delta: float,
    processing_time_ms: float,
    input_size: int,
    output_size: int,
    quantum_enhancement_factor: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log privacy transformation operation"""
    try:
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO privacy_transforms (
                    transform_type, privacy_budget_epsilon, privacy_budget_delta,
                    processing_time_ms, input_size, output_size,
                    quantum_enhancement_factor, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, transform_type, privacy_budget_epsilon, privacy_budget_delta,
                 processing_time_ms, input_size, output_size,
                 quantum_enhancement_factor, json.dumps(metadata) if metadata else None)
    except Exception as e:
        logger.error(f"Failed to log privacy transform: {e}")

async def log_federated_training(
    round_number: int,
    participating_agents: int,
    validated_agents: int,
    validation_accuracy: Optional[float] = None,
    validation_loss: Optional[float] = None,
    privacy_budget_epsilon: Optional[float] = None,
    processing_time_sec: Optional[float] = None,
    aggregation_method: Optional[str] = None,
    blockchain_validation: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log federated training round"""
    try:
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO federated_training (
                    round_number, participating_agents, validated_agents,
                    validation_accuracy, validation_loss, privacy_budget_epsilon,
                    processing_time_sec, aggregation_method, blockchain_validation, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, round_number, participating_agents, validated_agents,
                 validation_accuracy, validation_loss, privacy_budget_epsilon,
                 processing_time_sec, aggregation_method, blockchain_validation,
                 json.dumps(metadata) if metadata else None)
    except Exception as e:
        logger.error(f"Failed to log federated training: {e}")

async def log_system_metric(
    component: str,
    metric_name: str,
    metric_value: float,
    metric_unit: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log system metric"""
    try:
        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO system_metrics (
                    component, metric_name, metric_value, metric_unit, metadata
                ) VALUES ($1, $2, $3, $4, $5)
            """, component, metric_name, metric_value, metric_unit,
                 json.dumps(metadata) if metadata else None)
    except Exception as e:
        logger.error(f"Failed to log system metric: {e}")

async def get_inference_stats(
    hours: int = 24,
    operation_type: Optional[str] = None
) -> Dict[str, Any]:
    """Get inference statistics for the last N hours"""
    try:
        async with get_connection() as conn:
            query = """
                SELECT 
                    COUNT(*) as total_operations,
                    AVG(processing_time_ms) as avg_processing_time,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(CASE WHEN quantum_enhanced THEN 1 END) as quantum_operations,
                    COUNT(CASE WHEN safety_fallback THEN 1 END) as safety_fallbacks
                FROM inference_logs 
                WHERE timestamp >= NOW() - INTERVAL '1 hour' * $1
            """
            
            params = [hours]
            if operation_type:
                query += " AND operation_type = $2"
                params.append(operation_type)
            
            row = await conn.fetchrow(query, *params)
            
            return {
                'total_operations': row['total_operations'],
                'avg_processing_time_ms': float(row['avg_processing_time']) if row['avg_processing_time'] else 0,
                'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0,
                'quantum_operations': row['quantum_operations'],
                'safety_fallbacks': row['safety_fallbacks']
            }
            
    except Exception as e:
        logger.error(f"Failed to get inference stats: {e}")
        return {}

async def get_privacy_transform_stats(hours: int = 24) -> Dict[str, Any]:
    """Get privacy transformation statistics for the last N hours"""
    try:
        async with get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_transforms,
                    AVG(privacy_budget_epsilon) as avg_epsilon,
                    AVG(privacy_budget_delta) as avg_delta,
                    AVG(processing_time_ms) as avg_processing_time,
                    AVG(quantum_enhancement_factor) as avg_quantum_factor
                FROM privacy_transforms 
                WHERE timestamp >= NOW() - INTERVAL '1 hour' * $1
            """, hours)
            
            return {
                'total_transforms': row['total_transforms'],
                'avg_epsilon': float(row['avg_epsilon']) if row['avg_epsilon'] else 0,
                'avg_delta': float(row['avg_delta']) if row['avg_delta'] else 0,
                'avg_processing_time_ms': float(row['avg_processing_time']) if row['avg_processing_time'] else 0,
                'avg_quantum_factor': float(row['avg_quantum_factor']) if row['avg_quantum_factor'] else 0
            }
            
    except Exception as e:
        logger.error(f"Failed to get privacy transform stats: {e}")
        return {}

async def get_federated_training_stats(rounds: int = 10) -> Dict[str, Any]:
    """Get federated training statistics for the last N rounds"""
    try:
        async with get_connection() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_rounds,
                    AVG(participating_agents) as avg_participants,
                    AVG(validation_accuracy) as avg_accuracy,
                    AVG(validation_loss) as avg_loss,
                    AVG(privacy_budget_epsilon) as avg_epsilon
                FROM federated_training 
                ORDER BY round_number DESC 
                LIMIT $1
            """, rounds)
            
            return {
                'total_rounds': row['total_rounds'],
                'avg_participants': float(row['avg_participants']) if row['avg_participants'] else 0,
                'avg_accuracy': float(row['avg_accuracy']) if row['avg_accuracy'] else 0,
                'avg_loss': float(row['avg_loss']) if row['avg_loss'] else 0,
                'avg_epsilon': float(row['avg_epsilon']) if row['avg_epsilon'] else 0
            }
            
    except Exception as e:
        logger.error(f"Failed to get federated training stats: {e}")
        return {}

async def health_check() -> Dict[str, Any]:
    """Database health check"""
    try:
        if not _pool:
            return {"status": "unhealthy", "error": "Database not initialized"}
        
        async with get_connection() as conn:
            await conn.execute('SELECT 1')
            
            # Get basic stats
            stats = await get_inference_stats(1)  # Last hour
            
            return {
                "status": "healthy",
                "connection_pool_size": _pool.get_size(),
                "free_connections": _pool.get_free_size(),
                "recent_operations": stats.get('total_operations', 0)
            }
            
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def cleanup_old_data(days: int = 30) -> int:
    """Clean up old data older than N days"""
    try:
        async with get_connection() as conn:
            # Delete old inference logs
            inference_deleted = await conn.execute("""
                DELETE FROM inference_logs 
                WHERE timestamp < NOW() - INTERVAL '1 day' * $1
            """, days)
            
            # Delete old privacy transforms
            privacy_deleted = await conn.execute("""
                DELETE FROM privacy_transforms 
                WHERE timestamp < NOW() - INTERVAL '1 day' * $1
            """, days)
            
            # Delete old federated training data
            training_deleted = await conn.execute("""
                DELETE FROM federated_training 
                WHERE timestamp < NOW() - INTERVAL '1 day' * $1
            """, days)
            
            # Delete old system metrics
            metrics_deleted = await conn.execute("""
                DELETE FROM system_metrics 
                WHERE timestamp < NOW() - INTERVAL '1 day' * $1
            """, days)
            
            total_deleted = sum([
                int(inference_deleted.split()[-1]),
                int(privacy_deleted.split()[-1]),
                int(training_deleted.split()[-1]),
                int(metrics_deleted.split()[-1])
            ])
            
            logger.info(f"Cleaned up {total_deleted} old records")
            return total_deleted
            
    except Exception as e:
        logger.error(f"Failed to cleanup old data: {e}")
        return 0
