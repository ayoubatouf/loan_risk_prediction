from setuptools import find_packages, setup
from pathlib import Path

requirements = Path("requirements.txt").read_text().splitlines()

setup(
    name="loan_risk_prediction",
    version="0.1.0",
    description="A machine learning project for predicting loan risk.",
    author="Ayoub Atouf",
    author_email="atouf.ayoub.1@gmail.com",
    url="https://github.com/ayoubatouf/loan_risk_prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
