from setuptools import setup, find_packages

setup(
    name='NeuroLogit',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "floras_helpers @ git+https://github.com/takacsflora/floras-helpers.git@main"
    ],
    author='Flora Takacs',
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

