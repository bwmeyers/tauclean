Getting Started
===============

Installation
~~~~~~~~~~~~

Using uv (recommended):

.. code-block:: bash

    uv pip install tauclean

Using pip:

.. code-block:: bash

    pip install tauclean

Basic Usage
~~~~~~~~~~~

The main deconvolution function is :func:`tauclean.clean.clean`. 
Here's a simple example:

.. code-block:: python

    import numpy as np
    from tauclean.clean import clean

    # Create or load a pulsar profile
    profile = np.array([...])  # your profile data

    # Run CLEAN deconvolution
    result = clean(
        data=profile,
        tau=1.0,  # scattering timescale in ms
        period=100.0,  # pulsar period in ms
        pbftype='thin',  # pulse broadening function type
        threshold=3.0,  # noise threshold for termination
    )

    # Access results
    clean_components = result['cc']
    reconstructed = result['reconstruction']

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

The package includes two command-line scripts:

**tauclean** - Perform deconvolution on pulsar data:

.. code-block:: bash

    tauclean --help

**simulate** - Generate simulated pulsar profiles:

.. code-block:: bash

    simulate --help

Parameters
~~~~~~~~~~

Key parameters for the :func:`tauclean.clean.clean` function:

- **data** (np.ndarray): The observed pulse profile
- **tau** (float): Scattering timescale in milliseconds
- **period** (float): Pulsar spin period in milliseconds (default: 100.0)
- **pbftype** (str): Type of pulse broadening function to use:
  
  - ``'thin'`` - Thin screen scattering model
  - ``'thick'`` - Thick screen scattering model
  - ``'uniform'`` - Uniform scattering medium
  - ``'thick_exp'`` - Thick screen with exponential
  - ``'uniform_exp'`` - Uniform with exponential

- **gain** (float): Loop gain for component scaling (default: 0.05)
- **threshold** (float): Noise threshold for termination (default: 3.0)
- **iter_limit** (int): Maximum iterations (default: 1000)
- **onpulse_estimator** (str or list): On-pulse region definition:
  
  - ``'auto'`` - Automatically determine on/off-pulse regions
  - A list with bin indices for custom regions

Returns
~~~~~~~

The :func:`tauclean.clean.clean` function returns a dictionary containing:

- ``'cc'`` - Clean component positions and amplitudes
- ``'reconstruction'`` - Reconstructed intrinsic profile
- ``'fom'`` - Figures of merit for each iteration
- ``'n_iter'`` - Number of iterations performed
- And other analysis metrics

Advanced Usage
~~~~~~~~~~~~~~

For more control, you can provide pre-computed restoring and instrumental 
response functions:

.. code-block:: python

    from tauclean.clean import clean, get_restoring_function, get_inst_resp

    # Pre-compute functions
    inst_resp, inst_width = get_inst_resp(
        profile=profile,
        pulse_period=100.0,
        r_dm_width=0.1,
        r_pb_width=0.05,
        r_av_width=0.0,
        r_pd_width=0.0,
    )

    rest_func = get_restoring_function(
        profile=profile,
        pulse_period=100.0,
        inst_resp_width=inst_width,
    )

    # Run with custom functions
    result = clean(
        data=profile,
        tau=1.0,
        period=100.0,
        rest_func=rest_func,
        inst_resp_func=inst_resp,
    )
