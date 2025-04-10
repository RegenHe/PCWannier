from setuptools import setup, find_packages

setup(
    name='PCWannier',
    version='0.1.0',
    author='Yu He',
    description='',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'PCWannier=main:main',
        ],
    },
    python_requires='>=3.7',
)
