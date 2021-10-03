"""
EO-Forge package
"""

# Set the version number from VERSION file generated during the package installation
from pathlib import Path
PACKAGE_ROOT_DIR = Path(__file__).parent

with open(PACKAGE_ROOT_DIR/'VERSION', 'r') as version_file:
    __version__ = version_file.readline()
    version = __version__
