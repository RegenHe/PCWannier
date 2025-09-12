from setuptools import setup, find_packages

setup(
    name='PCWannier',
    version='0.1.1',
    author='Yu He',
    description='',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        'matplotlib==3.10.1',
        'numpy==2.2.4',
        'scipy==1.15.2',
        'setuptools==75.8.1',
        'threadpoolctl==3.6.0',
        'spglib==2.6.0',
        'spgrep==0.3.11',
        'cloudpickle==3.1.1',
    ],
    entry_points={
        'console_scripts': [
            'PCWannier=PCWannier.main:main',
        ],
    },
    python_requires='>=3.7',
)
