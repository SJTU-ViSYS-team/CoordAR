# Standard Library
import os
from pathlib import Path

# Third Party
from joblib import Memory


CACHE_DIR = "./.cache/megapose"

MEMORY = Memory(CACHE_DIR, verbose=0)
