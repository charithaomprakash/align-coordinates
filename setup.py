from setuptools import setup, find_packages

setup(
    name="align?coordinates",
    version='1.0',
    packages=find_packages(),
    entry_points={"console_scripts": "align_coordinates = align_coordinates:main"},
    author="C.Omprakash",
    description="align_coordinates",
    url="https://https://github.com/charithaomprakash/align_coordinates",
    setup_requires=[
        "pytest",
    ],	
    install_requires=[
        "pytest-shutil",
        "numpy",
        "matplotlib",
        "pathlib",
        "pandas",
        "ruamel.yaml"
    ],
)
