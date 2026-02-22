from setuptools import setup, find_packages

setup(
    name="semiconductor-test-toolkit",
    version="1.0.0",
    author="Mst Arefin Aktar",
    author_email="mst.arefinaktar02@gmail.com",
    description="Professional Python toolkit for IC Test Engineers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mstarefinaktar/semiconductor-test-toolkit",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "seaborn>=0.12",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
)
