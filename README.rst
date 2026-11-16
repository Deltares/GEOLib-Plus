GEOLib+
=============================

GEOLib+ is a Python package to read, interprent and plot cpt files.
The package can also be used to get soil parameters for constitutive models.

Installation
------------

Install GEOLib+ with:

.. code-block:: bash

    $ pip install d-geolib-plus


Requirements
------------

GEOLib+ requires Python 3.12 or newer. The dependencies are installed
automatically by pip. To install from a clone of the repository, run:

.. code-block:: bash

    $ pip install -e .


Testing & Development
---------------------

Install the development dependencies:

.. code-block:: bash

    $ pip install -e ".[dev]"

In order to run the testcode, from the root of the repository, run:

.. code-block:: bash

    $ pytest

Running flake8, mypy is also recommended. For mypy use:

.. code-block:: bash

    $ mypy --config-file mypy.ini geolib_plus


Documentation
-------------

In order to run the documentation, from the root of the repository, run:

.. code-block:: bash

    $ pip install -e ".[docs]"
    $ cd docs
    $ sphinx-build . build -b html -c .


The documentation is now in the `build` subfolder, where you can open 
the `index.html` in your browser.

Build wheel
-----------

To build a distributable wheel package, run:

.. code-block:: bash

    $ pip install build
    $ python -m build

The distributable packages are now built in the `dist` subfolder.