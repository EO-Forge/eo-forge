# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="eo-forge",
    version="0.1",
    author="EO-Forge team.",
    packages=find_packages(),
    license="LICENSE",
    include_package_data=True,
    description="Python tools for remote sensing data processing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/x-rst",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
