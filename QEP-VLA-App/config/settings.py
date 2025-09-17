"""
Configuration settings for QEP-VLA Platform
"""

import os
from typing import Dict, Any

def get_settings() -> Dict[str, Any]:
    """Get application settings"""
    return {
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'device': os.getenv('DEVICE', 'cpu'),
        'privacy_budget': float(os.getenv('PRIVACY_BUDGET', '0.1')),
        'blockchain_enabled': os.getenv('BLOCKCHAIN_ENABLED', 'True').lower() == 'true',
        'quantum_enhancement': os.getenv('QUANTUM_ENHANCEMENT', 'True').lower() == 'true'
    }
