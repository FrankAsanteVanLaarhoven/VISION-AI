#!/usr/bin/env python3
"""
Run PVLA Navigation System Web Application
Simple script to start the web server and open the browser
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def run_web_server():
    """Run the web server"""
    print("🚀 Starting PVLA Navigation System Web Server...")
    
    # Change to the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Add the project root to Python path
    sys.path.insert(0, str(project_root))
    
    try:
        # Import and run the web server
        from src.web_server import app
        import uvicorn
        
        print("✅ PVLA system components loaded successfully")
        print("🌐 Starting web server on http://localhost:8000")
        print("📱 Open your browser to http://localhost:8000")
        print("🔄 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Error importing PVLA components: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        return False

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open("http://localhost:8000")
        print("🌐 Browser opened automatically")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("🌐 Please open http://localhost:8000 manually")

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 PVLA Navigation System - Web Application")
    print("   Privacy-Preserving Vision-Language-Action Navigation")
    print("=" * 60)
    
    # Start browser opening in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the web server
    run_web_server()

if __name__ == "__main__":
    main()
