"""
Main application launcher for Organic Farm Pest Management AI System
Clean entry point that launches the Gradio interface
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from presentation.pest_interface import create_pest_management_interface
from config.config import GRADIO_CONFIG

def main():
    """
    Launch the pest management interface
    """
    print(f"🚀 Starting {GRADIO_CONFIG['title']}...")
    print(f"🌐 Server will run on port {GRADIO_CONFIG['server_port']}")
    
    # Create and launch the interface
    interface = create_pest_management_interface()
    
    interface.launch(
        server_port=GRADIO_CONFIG['server_port'],
        server_name=GRADIO_CONFIG['server_name'],
        share=GRADIO_CONFIG['share'],
        debug=GRADIO_CONFIG['debug']
    )

if __name__ == "__main__":
    main()
