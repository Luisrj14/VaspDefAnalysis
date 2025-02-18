from setuptools import setup, find_packages

# Read the README file for the long description (optional)
with open("README.md") as file:
    long_description = file.read()

setup(
    name="VaspDefAnalisys",                                                # Name of your package
    version="1.0",                                                         # Initial version of your package
    description="Analysis tools for VASP defect calculations",             # Short description
    long_description=long_description,                                     # Read from README file
    long_description_content_type="text/markdown",                         # Format of README (markdown)
    author="Luis Jimenez A",                                               # Author of the project
    #author_email="your.email@example.com",                                 # Author email
    url="https://github.com/luisrj11/VaspDefAnalysis",                     # URL of your project (if hosted on GitHub)
    #packages=find_packages(exclude=["bin"]),                              
    packages=find_packages(),                                              # Automatically find all Python packages                             
    install_requires=[  # Required dependencies
        "numpy",  # Example: Adding numpy as a dependency
        "matplotlib",  # Adding matplotlib as a dependency
        "git+https://github.com/QijingZheng/VaspBandUnfolding",  # Adding GitHub repository as dependency
        "ase",  # Adding ASE package for atomic simulations
    ],
    classifiers=[  # Classifiers that help categorize the package
        "Programming Language :: Python :: 3",  # This works with Python 3
        "Programming Language :: Python :: 3.10",  # Specifically Python 3.10
        "License :: OSI Approved :: MIT License",  # MIT License
        "Operating System :: OS Independent",  # Works on any OS
    ],
    python_requires=">=3.10",  # Minimum Python version required (3.10 and later)
)