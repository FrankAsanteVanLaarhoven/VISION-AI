#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    print("Testing imports...")
    
    # Test settings import
    from config.settings import get_settings
    print("✅ Settings import successful")
    
    # Test getting settings
    settings = get_settings()
    print(f"✅ Settings object created: {settings.app_name}")
    
    print("✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
