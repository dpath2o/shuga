#!/usr/bin/env python3
"""CLI wrapper for downloading and processing OSI-SAF-450 SIA."""
import sys
from pathlib import Path
repo_root = Path.home() / "AFIM" / "src" / "mawsons-chest"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from shuga.observations.OSI_SAF450 import main

if __name__ == "__main__":
    main()
