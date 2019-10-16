=========
Reasoners
=========

Reasoners must be configured by placing Python modules implementing the
:class:`.Reasoner` interface into the `evowluator/user/reasoners` directory.

Output format
=============

By default, evOWLuator expects reasoners to print certain information to standard output and/or
to file, as detailed in the following. In case it is not possible to control the output format
of a certain reasoner, it is possible to override the :meth:`~.Reasoner.results_parser` property,
returning a suitable subclass of the :class:`.ResultsParser` base class.

Classification
--------------

The reasoner must print the following to standard output:

.. code-block:: text

    Parsing: <float> ms
    Reasoning: <float> ms

When invoked for correctness evaluation, the reasoner must output the inferred taxonomy as
an OWL ontology in one of the syntaxes supported by evOWLuator (|syntaxes|). Specifically,
the ontology must contain all the direct (told and inferred) `SubClassOf` and `EquivalentClasses`
axioms, starting from `owl:Thing`. Axioms having `owl:Nothing` as the subclass can be omitted.

Consistency
-----------

The reasoner must print the following to standard output:

.. code-block:: text

    Parsing: <float> ms
    Reasoning: <float> ms
    The ontology is [not] consistent.

Matchmaking
-----------

The reasoner must print the following to standard output:

.. code-block:: text

    Resource parsing: <float> ms
    Request parsing: <float> ms
    Reasoner initialization: <float> ms
    Reasoning: <float> ms

When invoked for correctness evaluation, the reasoner must output matchmaking results as a
text file. The format of said file must match that of the reasoner used as a reference.

Java SE integration
===================

The :class:`.JavaReasoner` template allows integrating reasoners compiled into `jar` files.
In this case, :meth:`~.Reasoner.path` is the path to the jar file, and :meth:`~.Reasoner.args`
returns the arguments that should be passed to it. Flags that should be passed to the
Java Virtual Machine can be specified using the :meth:`~.JavaReasoner.vm_opts` property.

Mobile integration
==================

Android
-------

The :class:`.AndroidReasoner` template allows the framework to run reasoning tasks
on Android devices. Reasoners can be easily integrated via *Gradle*, either as module dependencies
or external libraries. Each reasoner must be wrapped in an Android application declaring the
:code:`<reasoner app package>.EVOWLUATE` intent_ in its `AndroidManifest.xml` file.
evOWLuator invokes the intent and passes the following data via its `extras`, which can be
recovered by calling the :code:`Bundle.getString()` method:

- :code:`task`: requested inference task.
- :code:`resource`: name of the input ontology.
- :code:`request`: only for matchmaking tasks, name of the request ontology.

Output must be logged via logcat_, and each message must be prefixed with a unique string,
which is used by the framework to filter `logcat` output. Datasets can be either uploaded to
the device before running the evaluation, or retrieved programmatically by the application.

evOWLuator automatically installs a *launcher* application containing an implementation of the
Instrumentation_ class, whose purpose is to invoke the reasoners and to close wrapper apps
once the reasoning task is complete. The framework communicates with the launcher via ADB_,
therefore *USB debugging* must be enabled via the Settings app of the target device.

Once the reasoning task is complete, the reasoner application must invoke the
:code:`it.poliba.sisinflab.owl.evowluator.END` intent.

With regards to the Python part of the integration, the :meth:`~.Reasoner.args` method
must not be overridden. Instead, one has to override the properties and methods indicated in the
:class:`.AndroidReasoner` docs, which the template uses to build the argument vector appropriately
before invoking `adb`.

iOS
---

The :class:`.IOSReasoner` template class enables running reasoning tasks on iOS mobile devices.
Reasoners must be implemented in separate Xcode_ projects, and specifically as Xcode test cases,
i.e. :code:`XCTestCase` subclasses. They can be integrated either by including their source files,
or by linking them with the test target, if they are available as libraries.

Each supported reasoning task must be wrapped in a dedicated method of the test case,
which has then to be deployed to the target device, together with datasets used for the evaluation.
The latter can be uploaded via the *copy bundle resources* build phase, though this is not
strictly necessary: the app can retrieve the necessary ontologies however it pleases.

evOWLuator invokes test cases through `xcodebuild`, Xcode's command line interface, passing the
following data via environment variables, which can be accessed via the :code:`ProcessInfo` class:

- :code:`RESOURCE`: name of the input ontology.
- :code:`REQUEST`: only for matchmaking tasks, name of the request ontology.

Reasoner output must have the same format described earlier, and it must be printed
to standard output.

Similarly to how Android integration is handled, the :meth:`~.Reasoner.args` method
must not be overridden. Instead, one has to override the properties and methods indicated in the
:class:`.IOSReasoner` docs, which the template uses to appropriately construct `xcodebuild`'s
arguments.

.. _ADB: https://developer.android.com/studio/command-line/adb
.. _intent: https://developer.android.com/reference/android/content/Intent
.. _Instrumentation: https://developer.android.com/reference/android/app/Instrumentation
.. _logcat: https://developer.android.com/studio/debug/am-logcat
.. _Xcode: https://developer.apple.com/xcode
