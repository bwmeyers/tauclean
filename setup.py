from setuptools import setup

reqs = ['numpy']

setup(
    name='tauclean',
    version='1.0',
    url='https://github.com/bwmeyers/tauclean',
    license='',
    author='Bradley Meyers',
    author_email='bradley.meyers1993@gmail.com',
    description='A package to deconvolve scattered pulsar profiles',
    install_requires=reqs,
    scripts=['tauclean.py', 'simulate.py']
)
