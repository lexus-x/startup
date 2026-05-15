from setuptools import setup, find_packages

setup(
    name="ksam",
    version="0.2.0",
    description="Kinematic Singularity Awareness Module for VLA models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
