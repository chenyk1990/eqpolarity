#!/usr/bin/env python
# -*- encoding: utf8 -*-
# import glob
# import inspect
# import io
# import os

from setuptools import setup
long_description = """
Source code: https://github.com/chenyk1990/eqpolarity""".strip() 


def read(*names, **kwargs):
    return io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")).read()

setup(
    name="eqpolarity",
    version="0.0.0.3",
    license='MIT License',
    description="EQpolarity package is a deep-learning-based package for determining earthquake first-motion polarity",
    long_description=long_description,
    author="eqpolarity developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/chenyk1990/eqpolarity",
    packages=['eqpolarity'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "seismology", "earthquake seismology", "exploration seismology", "array seismology", "science", "engineering", "artificial intelligence", "deep learning"
    ],
    install_requires=[
        "numpy", 
        "scipy", 
        "matplotlib==3.8.0",
        "tqdm",
        "obspy",
        "tensorflow==2.14.0",
        "scikit-learn==1.2.2",
        "seaborn==0.13.2"
    ],
    python_requires='==3.11.7',
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
)
