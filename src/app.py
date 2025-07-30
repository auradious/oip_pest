"""
Legacy app launcher - DEPRECATED
Use presentation/app.py instead for the new modular structure
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from presentation.app import main

if __name__ == "__main__":
    print("⚠️  This file is deprecated. Use 'python presentation/app.py' instead.")
    print("🔄 Redirecting to new modular structure...")
    main()
