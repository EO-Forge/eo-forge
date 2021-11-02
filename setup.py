import setuptools
import os
import configparser

from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).parent

if __name__ == "__main__":
    setup_cfg_path = os.path.join(os.path.dirname(__file__), "setup.cfg")

    config = configparser.ConfigParser()
    config.read(setup_cfg_path)
    version = config["metadata"]["version"]

    with open(PROJECT_ROOT_DIR / "eo_forge/VERSION", "w") as version_file:
        version_file.write(version)

    setuptools.setup()
