=============
What is this?
=============

*evOWLuator* is cross-platform, energy aware evaluation tool for OWL reasoners,
developed by `SisInf Lab`_ at the `Polytechnic University of Bari`_.

Motivation
==========

Research in the field of Semantic Web technologies and development of reasoning systems requires
rigorous evaluation methodologies, benchmarks and software frameworks. This is particularly crucial
when designing new systems, or repurposing existing ones, for **ubiquitous computing** scenarios.
The deployment of Semantic Web reasoners on mobile and embedded devices introduces concerns
regarding the **energy usage** of such systems, which was pretty much a non-issue for
desktop-oriented ones.

*evOWLuator* easens the assessment of inference tasks implemented by OWL reasoners, while being
sufficiently flexible to support running reasoning tasks on remote devices, with *Android* and *iOS*
supported out-of-the-box. Its plugin architecture makes it very extensible and configurable,
especially with regards to integrating reasoning systems, supporing additional target platforms
and interfacing with energy profilers and sensors.

Features
========

evOWLuator currently supports the assessment of the **ontology classification**
and **consistency** standard reasoning tasks, as well as the **matchmaking**
non-standard reasoning task. Inference services can be tested under different **evaluation modes**:

- **Correctness**: verifies that results of an inference task match those of a reference reasoner.
- **Performance**: measures turnaround time and peak memory usage.
- **Energy footprint**: estimates the energy usage of the reasoner.

The system is also able to generate **summaries** and **graphical visualizations**, providing
human-understandable representations of the raw data collected during previous evaluations.

License
=======

evOWLuator is distributed under the `Eclipse Public License, Version 2.0`_.

.. _Eclipse Public License, Version 2.0: https://www.eclipse.org/legal/epl-2.0
.. _Polytechnic University of Bari: http://www.poliba.it
.. _SisInf Lab: http://sisinflab.poliba.it/swottools
