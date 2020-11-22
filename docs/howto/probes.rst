=============
Energy probes
=============

Modules implementing energy probes (via the :class:`.EnergyProbe` interface) must be positioned
in the `evowluator/user/probes` directory. This last step is only necessary
for energy footprint evaluation. Note that evOWLuator comes with two built-in probes that wrap
the `powermetrics` and `powertop` tools on macOS and GNU/Linux, respectively.

evOWLuator estimates energy consumption by running the reasoner binary and polling the probe
until the process exits. The collected samples are then used to compute an *energy footprint*
score as follows:

.. math::

   score = sampling\_interval * \sum_{i=0}^N sample_i

This provides a proxy of the energy used by the reasoner during its execution time, which can be
leveraged for comparisons among reasoners run on the same device and using the same probe.
