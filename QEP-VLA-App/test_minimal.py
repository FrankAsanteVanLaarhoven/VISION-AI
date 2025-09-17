#!/usr/bin/env python3
"""
Minimal test script to isolate the import issue
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    
    # Test settings import
    print("1. Testing settings import...")
    from config.settings import get_settings
    print("   ✅ Settings import successful")
    
    # Test core imports
    print("2. Testing core imports...")
    from core.quantum_privacy_transform import QuantumPrivacyTransform
    print("   ✅ QuantumPrivacyTransform import successful")
    
    from core.federated_trainer import SecureFederatedTrainer
    print("   ✅ SecureFederatedTrainer import successful")
    
    from core.edge_inference import AdaptiveEdgeInferenceEngine
    print("   ✅ AdaptiveEdgeInferenceEngine import successful")
    
    # Test app import
    print("3. Testing app import...")
    from app import app
    print("   ✅ App import successful")
    
    print("\n🎉 All imports successful! The app should work.")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
