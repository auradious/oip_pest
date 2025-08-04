#!/usr/bin/env python3
"""
Launcher for the Conversational Pest Management AI System
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the conversational pest management interface"""
    print("🌱 Starting Conversational Pest Management AI System...")
    print("=" * 60)
    
    try:
        # Import and create the interface
        from presentation.conversational_interface import create_conversational_pest_interface
        
        print("✅ Loading AI models and services...")
        interface = create_conversational_pest_interface()
        
        print("🚀 Launching interface...")
        print("📱 Access the interface at: http://localhost:7860")
        print("💬 Features available:")
        print("   • 📸 Pest identification from images")
        print("   • 🤖 Conversational AI consultant")
        print("   • 🌿 Organic treatment recommendations")
        print("   • 💡 Quick question buttons")
        print("   • 📄 Chat history export")
        print("=" * 60)
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        print("💡 Check that:")
        print("   • All model files are present in ./models/")
        print("   • Ollama is running (if using conversational features)")
        print("   • Port 7860 is available")
        sys.exit(1)

if __name__ == "__main__":
    main()