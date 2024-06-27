import setuptools

setuptools.setup()

# from setuptools import setup
# from os import path
# import sys
# import re


# def get_version():
#     here = path.abspath(path.dirname(__file__))
#     with open(path.join(here, "tauclean/__init__.py")) as f:
#         contents = f.read()

#     return re.search(r"__version__ = \"(\S+)\"", contents).group(1)


# reqs = ['numpy>=1.16.0', 'matplotlib>=3.0.0', 'scipy>=1.2.0']

# setup(
#     name='tauclean',
#     version=get_version(),
#     url='https://github.com/bwmeyers/tauclean',
#     author='Bradley Meyers',
#     author_email='bradley.meyers1993@gmail.com',
#     license='AFL-3.0',
#     description='A package to deconvolve scattered pulsar profiles',
#     keywords="signal processing",
#     install_requires=reqs,
#     packages=['tauclean'],
#     scripts=['scripts/tauclean', 'scripts/simulate'],
#     setup_requires=['pytest-runner'],
#     tests_require=['pytest', 'nose']
# )
