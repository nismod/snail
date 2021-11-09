#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Setup snail package
"""
from glob import glob

# pybind11 will be available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext

from setuptools import find_packages, setup


ext_modules = [
    Pybind11Extension(
        "snail.core.intersections",
        sorted(glob("src/cpp/*.cpp")),
    ),
]


def readme():
    """Read README contents
    """
    with open('README.md', encoding='utf8') as f:
        return f.read()


setup(
    name='nismod-snail',
    use_scm_version=True,
    license='MIT License',
    description='The spatial networks impact assessment library',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Tom Russell',
    author_email='tomalrussell@gmail.com',
    url='https://github.com/nismod/snail',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Utilities',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires=">=3.6",
    install_requires=[
        'geopandas', 'shapely', 'rasterio', 'python-igraph'
    ],
    extras_require={
        'dev': ['affine', 'black', 'nbstripout', 'numpy', 'pytest', 'pytest-cov'],
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'snail_split = snail.cli:snail_split',
            'snail_raster2split = snail.cli:snail_raster2split',
        ]
    },
)
