from setuptools import setup, find_packages

setup(
    name='NeuroLogit',
    version='1.0',
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly",
        "pyarrow",
        "jupyter",
        "floras_helpers @ git+https://github.com/takacsflora/floras-helpers.git@main",
    ],
    author='Flora Takacs',
    author_email='takacsflora@gmail.com',
    description='Logistic Classification for ephys data and optogentic inactivation',
    long_description='',
    license='MIT',
    url='https://github.com/takacsflora/NeuroLogit',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)

