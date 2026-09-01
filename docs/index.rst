tauclean Documentation
======================

A pulsar profile deconvolution method for the recovery of intrinsic profile 
shapes and ISM broadening functions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   references

About
-----

**tauclean** is a Python implementation of the CLEAN deconvolution algorithm 
for analyzing pulsar profiles. It recovers intrinsic pulse shapes and 
interstellar medium (ISM) broadening functions from observed scatter-broadened 
pulsar signals.

Key Features
~~~~~~~~~~~~

- CLEAN deconvolution algorithm for pulsar profile analysis
- Multiple pulse broadening function models (thin, thick, uniform, etc.)
- Automated and manual on/off-pulse region estimation
- Comprehensive figures of merit for convergence analysis
- Flexible instrumental response modeling

Installation
~~~~~~~~~~~~

You can install tauclean using `uv`:

.. code-block:: bash

    uv pip install tauclean

Or with pip:

.. code-block:: bash

    pip install tauclean

Quick Start
~~~~~~~~~~~

Basic usage:

.. code-block:: python

    from tauclean import clean

    # Load your pulsar profile data
    profile = ...  # your profile data

    # Run CLEAN deconvolution
    result = clean(
        data=profile,
        tau=1.0,  # scattering timescale in ms
        period=100.0,  # pulsar period in ms
        pbftype='thin'
    )

Citation
--------

If you use this code, please cite the original papers:

- **Original Method**: Bhat et al. (2003), ApJ 584, 782-790
- **Modern Notebook implementation**: Young & Lam (2024), ApJ 962, 131

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
