from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="consciousnessx",
    version="0.1.0",
    author="Dafydd Napier",
    author_email="dafydd.napier@consciousnessx.ai",
    description="Quantum-Biological Consciousness Simulation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Napiersnotes/consciousnessX",
    project_urls={
        "Bug Tracker": "https://github.com/Napiersnotes/consciousnessX/issues",
        "Documentation": "https://consciousnessx.readthedocs.io/",
        "Source Code": "https://github.com/Napiersnotes/consciousnessX",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=11.0.0", "torch>=2.0.0"],
        "quantum": ["qutip>=4.7.0", "qiskit>=0.43.0"],
        "visualization": ["plotly>=5.14.0", "dash>=2.9.0"],
        "hpc": ["mpi4py>=3.1.0", "ray>=2.4.0"],
        "dev": [
            "pytest>=7.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "consciousnessx=consciousnessx.cli:main",
            "cx-simulate=consciousnessx.simulate:main",
            "cx-visualize=consciousnessx.visualize:main",
        ],
    },
    include_package_data=True,
    package_data={
        "consciousnessx": [
            "configs/*.yaml",
            "data/*.json",
            "models/*.pt",
        ],
    },
)
