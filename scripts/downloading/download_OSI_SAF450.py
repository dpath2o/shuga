#!/usr/bin/env python3
"""CLI wrapper for downloading and processing OSI-SAF-450 SIA."""
import sys
from pathlib import Path
package_parent = Path(__file__).resolve().parents[3]
if str(package_parent) not in sys.path:
    sys.path.insert(0, str(package_parent))
from shuga.observations.OSI_SAF450 import main

if __name__ == "__main__":
    main()
