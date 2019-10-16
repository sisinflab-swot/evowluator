============
Installation
============

evOWLuator can be used on **Linux**, **macOS** and **Windows** (via WSL_).
The following sections detail its requirements, setup, and some platform-specific instructions.

Requirements
============

In order to run evOWLuator you will need:

- git_ version 2.0 or later.
- Python_ version 3.7 or later.
- pip_ version 19.0 or later.
- Java_ version 10.0 or later.
- Gradle_ version 5.6 or later.

Ubuntu
------

.. code-block:: bash

   apt install default-jdk git gradle python3 python3-dev python3-pip python3-tk python3-venv

macOS
-----

It is recommended to install the required dependencies via HomeBrew_:

.. code-block:: bash

   brew install git gradle python3
   brew cask install java

Download and setup
==================

You can find evOWLuator's code on its `git repository <git_url_>`_. Please note that it contains
submodules, so it is recommended that you clone it using the `--recursive` flag.

.. code-block:: bash

   git clone --recursive <repo URL> <dir>

After cloning the repo you must run the `setup.sh` script, which fetches the remaining libraries,
sets up a virtual environment via venv_ and installs the necessary Python packages.

In general, the framework will use the global `Python` interpreter; if multiple versions of
the interpreter are installed, the user can specify which one will be used by evOWLuator by
exporting its path to the :code:`EVOWLUATOR_PYTHON` environment variable before running `setup.sh`.

Building the documentation
==========================

It is also possible to build the docs you are currently viewing by invoking the `build.sh` script
in the `docs` subdirectory. This will install Sphinx_ and other requirements in the
virtual environment created by the setup script, and it will output HTML docs in the
`docs/_build/html` subdirectory. Build logs are stored under `docs/_build/log`.

.. _git: https://git-scm.com
.. _Gradle: https://gradle.org
.. _HomeBrew: https://brew.sh
.. _Java: https://java.com
.. _Python: https://python.org
.. _pip: https://pypi.org/project/pip/
.. _Sphinx: http://sphinx-doc.org
.. _venv: https://docs.python.org/3/library/venv.html
.. _WSL: https://docs.microsoft.com/windows/wsl
