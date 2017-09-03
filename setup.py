#!/usr/bin/env python

from os.path import exists
from setuptools import setup


setup(name='dask_geopandas',
      version='0.0.1',
      description='Dask and GeoPandas',
      url='http://github.com/dask/dask-geopandas/',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='dask, geopandas',
      packages=['dask_geopandas'],
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      zip_safe=False)
