.. BIDS preprocessing documentation master file, created by
   sphinx-quickstart on Tue May 30 21:28:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Brain Pipe's documentation!
==============================================


.. _brain-pipe-figure:
.. figure:: /_images/brain_pipe_github.svg
    :align: center
    :width: 100%

    **Brain pipe** : Preprocess brain imaging datasets efficiently and extensibly in Python3.


Motivation
----------
This repository contains code to efficiently preprocess BIDS datasets in python3.
The initial main goal of this code is to use
it for the public EEG dataset of the `ICASSP 2023 Auditory EEG challenge <https://exporl.github.io/auditory-eeg-challenge-2023/>`_,
i.e. the `sparrKULee dataset <https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND>`_
and possibly further extend to other EEG/brain datasets where efficient preprocessing is necessary.

Currently, only EEG datasets are supported.

Internal structure
------------------

Most of the code relies on a pipeline structure which is explained in more detail in
:ref:`pipeline` and :ref:`configuration`

Installation
------------

You can install this repository as a `pip <https://pip.pypa.io/en/stable/>`_ package::

.. code-block:: bash

   $ pip install brain_pipe

Running
-------

You can create your own script to run the pipeline (see also :ref:`pipeline`), or you
can use the command line interface (see also :ref:`cli` and :ref:`configuration`) with a configuration file.

Requirements
------------

Python >= 3.7


Contributing
------------

Contributions are welcome. Please open an issue or a pull request.
For more information, see `CONTRIBUTING.md <CONTRIBUTING.md>`_.


API reference
-------------

See :ref:`api`


Go to
-----

.. toctree::
   :maxdepth: 1

   pipeline
   configuration
   cli
   api/api


.. toctree::
    :maxdepth: 1
    :hidden:

    api/full_tree