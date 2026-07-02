from setuptools import setup, find_packages

setup(
    name='PCWannier',
    version='0.1.2',
    author='Yu He',
    description='',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        'matplotlib>=3.10.1',
        'numpy>=2.2.4',
        'numba>=0.61.2',
        'scipy>=1.15.2',
        'setuptools>=75.8.1',
        'spglib>=2.6.0',
        'spgrep>=0.3.11',
    ],
    entry_points={
        'console_scripts': [
            'PCWannier=PCWannier.main:main',
        ],
    },
    python_requires='>=3.7',
)
