.. xcontour documentation master file

Welcome to xcontour's documentation!
====================================

.. image:: https://raw.githubusercontent.com/miniufo/xcontour/master/pics/sorting.jpg

This is a `Python` package built on `xarray`_, targeting at performing
diagnostic analyses and calculations in contour-based coordinates.  The new
coordinates are built based on iso-surfaces of any quasi-conservative tracers
(passive or active), mostly through a conservative rearrangement (or adiabatic
sorting in a particular direction).  Such rearrangement allows one to **isolate
the adiabatic advective process of a fluid and focus on non-conservative
processes**.

This project is `published on GitHub <https://github.com/miniufo/xcontour>`__
and can be cited using its `Zenodo DOI
<https://doi.org/10.5281/zenodo.473022002>`__.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Examples
   Contributors

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _xarray: http://xarray.pydata.org/en/stable/
