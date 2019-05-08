## Pre-requisites
Tested and installed on machines running Ubuntu 16.04 LTS and 18.04 LTS with Python versions 3.5, 3.6 and 3.7. 

Requires: numpy >= v1.16.0 (tested), scipy >= v1.2.0 (tested), matplotlib >= v3.0.0 (tested)

## Installing
To install, download the repository:

```bash
git clone https://github.com/bwmeyers/tauclean.git
```

Change into the downloaded directory (`tauclean`) and run:

```bash
pip install .
```

or

```bash
python setup.py install
```

Assuming you have the correct permissions, the module is now installed and the `tauclean` and `simulate` scripts should
be available on your PATH.
