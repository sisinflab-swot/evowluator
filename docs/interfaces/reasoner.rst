========
Reasoner
========

.. contents:: :local:

Base
====

.. autoclass:: evowluator.reasoner.base.Reasoner
   :members: name, path, args, supported_syntaxes, supported_tasks, preferred_syntax,
             setup, teardown, results_parser, is_template

Enumerations
------------

.. autoclass:: evowluator.reasoner.base.ReasoningTask
   :members: CLASSIFICATION, CONSISTENCY, MATCHMAKING, standard

.. autoclass:: evowluator.reasoner.base.MetaArgs
   :members: INPUT, OUTPUT, REQUEST

.. autoclass:: evowluator.evaluation.mode.EvaluationMode
   :members:

.. autoclass:: evowluator.data.ontology.Syntax
   :members:

Results
-------

.. autoclass:: evowluator.reasoner.results.ResultsParser
   :members:

.. autoclass:: evowluator.reasoner.results.ReasoningResults
   :members:

.. autoclass:: evowluator.reasoner.results.StandardReasoningResults
   :show-inheritance:

.. autoclass:: evowluator.reasoner.results.MatchmakingResults
   :show-inheritance:
   :members: init_ms, matchmaking_ms

.. autoclass:: evowluator.reasoner.results.EnergyStats
   :members:

Java SE
=======

.. autoclass:: evowluator.reasoner.java.JavaReasoner
   :show-inheritance:
   :members: vm_opts

Mobile
======

.. autoclass:: evowluator.reasoner.mobile.MobileReasoner
   :show-inheritance:

Android
-------

.. autoclass:: evowluator.reasoner.mobile.AndroidReasoner
   :show-inheritance:
   :members: target_package, log_prefix

iOS
---

.. autoclass:: evowluator.reasoner.mobile.IOSReasoner
   :show-inheritance:
   :members: project, scheme, test_name_for_task
