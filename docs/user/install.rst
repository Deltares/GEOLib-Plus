.. _install:

Installation
========================

This part of the documentation covers the installation of GEOLib+.
The first step to using any software package is getting it properly installed.

GEOLib+ installation
--------------------
GEOLib+ releases are available from publicwiki.deltares.nl as .whl files. You can
download these and install such files with::

    $ pip install  geolib_plus-0.1.1-py3-none-any.whl

To install the latest GEOLib+ simply the following command::

    $ pip install git+git@bitbucket.org:DeltaresGEO/geolib-plus.git

Note that both locations are private and require authentication.

.. warning::

    Note that installation of the GEOLib+ package and GEOLib with a python version
    newer than 3.9 might lead to package conflicts between the two libraries.
    And might require the reinstallation of a  few of those packages.
    This problem is using caused by version 1.6.2 of the pydantic package which is not compatible with 3.9 Python version.
    Upgrading to pydantic version 1.8.2 seems to solve this problem.

Packages used
-------------

The main packages used are:

- Lxml_ Powerful and Pythonic XML processing library combining libxml2/libxslt with the ElementTree API.
- Matplotlib_ Python plotting package
- Numpy_ NumPy is the fundamental package for array computing with Python.
- Pandas_ Powerful data structures for data analysis, time series, and statistics
- Poetry_ for package management (replacing setuptools) see also `PEP 518 <https://www.python.org/dev/peps/pep-0518/>`_.
- Pydantic_ for validation of types and some parameters (min/max/defaults).
- Pyproj_ Python interface to PROJ (cartographic projections and coordinate transformations library)
- Scipy_ SciPy: Scientific Library for Python
- Shapely_ Geometric objects, predicates, and operations


.. _Lxml: https://lxml.de/
.. _Matplotlib: https://matplotlib.org/
.. _Numpy: https://numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _Poetry: https://python-poetry.org/docs/
.. _Pydantic: https://pydantic-docs.helpmanual.io/
.. _Pyproj: https://pyproj4.github.io/pyproj/stable/
.. _Scipy: https://www.scipy.org/
.. _Shapely: https://shapely.readthedocs.io/en/stable/manual.html


You don't need to install anything manually, as the pip installation should take care of it.

Get the Source Code
-------------------

GEOLib+ is actively developed on BitBucket, where the code is
`always available <https://bitbucket.org/DeltaresGEO/geolib-plus/src>`_.

You can either clone the public repository::

    $ git clone git@bitbucket.org:DeltaresGEO/geolib-plus.git

Once you have a copy of the source, you can embed it in your own Python
package, or install it into your site-packages easily::

    $ cd geolib-plus
    $ pip install poetry
    $ poetry install

