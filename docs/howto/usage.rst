.. _usage:

=====
Usage
=====

evOWLuator can be invoked through the :code:`evowluate` binary. It is possible to get an overview
of available subcommands by invoking it with the :code:`-h` flag:

- :code:`classification`: runs the evaluation for the ontology classification task.
- :code:`consistency`: runs the evaluation for the ontology consistency task.
- :code:`matchmaking`: runs the evaluation for the matchmaking task.
- :code:`info`: prints information about configured reasoners and datasets.
- :code:`convert`: converts a dataset into the specified OWL syntax.
- :code:`visualize`: generates high-level statistics and plots for a previous evaluation.

All subcommands support the following help flags:

- :code:`-h, --help`: print help for the specified subcommand.
- :code:`--debug`: halt execution and print additional information on error.

Subcommand-specific flags are detailed in the following.

classification, consistency, matchmaking
========================================

**Required arguments:**

- :code:`-m, --mode <MODE>`: evaluation mode. Possible values:

  - :code:`correctness`: check the validity of reasoners' output, using one of them as an oracle.
    The test oracle is the first reasoner specified in the list following the :code:`-r` flag.
  - :code:`performance`: collect statistics about performance, in terms of turnaround time
    and maximum memory usage.
  - :code:`energy`: compute a relative estimate of the energy consumed by the inference task.
    This mode requires specifying the class name of the energy probe the framework should use
    via the :code:`-e` flag.

- :code:`-e, --energy-probe <PROBE>`: class name of the energy probe the framework should use, only
  required with :code:`-m energy`.

**Optional arguments:**

- :code:`-d, --dataset <DATASET>`: target dataset.
  **Default:** first dataset in the `data` directory.
- :code:`-r, --reasoners <REASONER> [<REASONER> ...]`: list of reasoners to use.
  **Default:** all configured reasoners.
- :code:`-n, --num-iterations <NUM>`: number of iterations for each test.
  **Default:** 5.
- :code:`-t, --timeout <TIMEOUT>`: timeout imposed on each reasoner for a single inference task,
  in seconds. **Default:** 1200.
- :code:`-s, --syntax <SYNTAX>`: reference OWL syntax.
  **Values:** |syntaxes|.
  **Default:** preferred syntax for each reasoner.
- :code:`--resume-after <ONTOLOGY>`: resume the evaluation after the specified ontology.

info
====

**Optional arguments:**

- :code:`-d, --dataset <DATASET>`: target dataset.
  **Default:** first dataset in the `data` directory.
- :code:`-r, --reasoners <REASONER> [<REASONER> ...]`: list of reasoners to use.
  **Default:** all configured reasoners.

convert
=======

**Required arguments:**

- :code:`-d, --dataset <DATASET>`: dataset to convert.
- :code:`-s, --syntax <SYNTAX>`: target syntax.
  **Values:** |syntaxes|

visualize
=========

**Required arguments:**

- :code:`path`: path to the directory containing the evaluation results to visualize.

**Optional arguments:**

- :code:`-s, --size <WIDTH> <HEIGHT>`: width and height of the figure in inches.
- :code:`-p, --plots <PLOT> [<PLOT> ...]`: subplots to show (**default:** all).
- :code:`-r, --reasoners <REASONER> [<REASONER ...]`: reasoners whose results should be plotted.
  **Default:** all configured reasoners.
- :code:`--no-gui`: do not show the interactive GUI.
- :code:`--no-titles`: omit titles for figures and axes.
- :code:`--no-labels`: omit value labels.
- :code:`--label-fmt <FORMAT>`: `float format`_ of value labels.
- :code:`--label-rot`: rotation of value labels in degrees.
- :code:`--xtick-rot`: rotation of labels on the x axis in degrees.
- :code:`--ytick-rot`: rotation of labels on the y axis in degrees.
- :code:`--legend-loc <LOCATION>`: location of the legend. **Values:**

  - :code:`none`: do not plot the legend.
  - :code:`best`: let the matplotlib engine decide the position.
  - :code:`[upper, lower, center] right, left, center`: explicit position.
- :code:`--legend-cols`: number of columns of the legend.
- :code:`--legend-only`: only plot the legend.
- :code:`--colors`: colors_ to use for each reasoner.
- :code:`--markers`: markers_ to use for each reasoner.
- :code:`--marker-size`: marker size in points.

.. _float format: https://docs.python.org/3/library/string.html#formatspec
.. _colors: https://matplotlib.org/tutorials/colors/colors.html
.. _markers: https://matplotlib.org/api/markers_api.html
