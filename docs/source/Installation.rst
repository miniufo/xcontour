Installation
============

Requirements
^^^^^^^^^^^^

xcontour is compatible with python 3 (>= version 3.8). It requires
xarray_, numpy_, numba_, xhistogram_, and scikit-image_.

Installation from pip
^^^^^^^^^^^^^^^^^^^^^

One can do this by using pip::

    pip install xcontour

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from github
^^^^^^^^^^^^^^^^^^^^^^^^

xcontour is still under active development. To obtain the latest development
version, you may clone the `source repository
<https://github.com/miniufo/xcontour>`_ and install it::

    git clone https://github.com/miniufo/xcontour.git
    cd xcontour
    python setup.py install

or simply::

    pip install git+https://github.com/miniufo/xcontour.git


How to run the notebooks
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run the example notebooks in this documentation, you will need
a few extra dependencies that you can install via::

    conda env create -f environment.yml
    conda activate xcontour


.. _xarray: http://xarray.pydata.org/en/stable/
.. _numpy: https://numpy.org/
.. _numba: https://numba.pydata.org/
.. _xhistogram: https://xgcm.github.io/xhistogram/
.. _scikit-image: https://scikit-image.org/
