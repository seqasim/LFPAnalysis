from setuptools import find_packages, setup

# Get the repository owner and name from the GitHub URL
github_url = 'https://github.com/seqasim/LFPAnalysis'

# Get long description
try:
    with open("README.md", "r", encoding='utf-8') as fh:
        __long_description__ = fh.read()
except FileNotFoundError:
    __long_description__ = 'Package to process LFP data'

# Get requirements
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        required = [line.strip() for line in f.read().splitlines() 
                    if line.strip() and not line.strip().startswith('#')]
except FileNotFoundError:
    required = []

setup(
    name='LFPAnalysis',
    version='1.0.0',
    description='Package to process LFP data',
    long_description=__long_description__,
    long_description_content_type='text/markdown',
    url=github_url,
    author='Salman Qasim',
    author_email='',  # Add email if desired
    packages=find_packages(),   
    package_data={'LFPAnalysis': ['YBA_ROI_labelled.xlsx']},
    include_package_data=True,
    install_requires=required,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

    