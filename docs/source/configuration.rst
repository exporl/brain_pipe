.. _configuration:

Defining and running Pipelines through configuration files
==========================================================

Instead of programmatically defining your Pipeline in a script, you can
define it through a configuration file. This allows you to run your
Pipeline in the same way you would run a script, but without having to
write any code.

Running pipelines from configuration files
------------------------------------------

You can run the pipeline in a configuration file by just using the :ref:`cli` :

.. code-block:: bash

    $ brain_pipe config_file.extension

Background
----------

A :class:`.Parser` is used to parse a configuration string into a list of tuples
of :class:`.DataLoader` and :class:`.Pipeline` objects. This list will then be provided
to a :class:`.Runner` to run all of the :class:`.Pipeline` s.

All parsers can be found in the :mod:`.parsers` module. The most important :class:`.Parser`
classes are:

1. :class:`.SimpleDictParser` - parses a dictionary of configuration options
   into a list of :class:`.DataLoader` and :class:`.Pipeline` objects. This is the
   base class for most other parsers.

2. :class:`.TextParser` - parses a string of configuration options into a list
   of :class:`.DataLoader` and :class:`.Pipeline` objects. This parser basically
   converts a text string into a dictionary and then uses the :class:`.SimpleDictParser`,
   from which it inherits, to parse the dictionary.

3. :class:`.FileParser` - parses a file of configuration options into a list of
   :class:`.DataLoader` and :class:`.Pipeline` objects. This parser basically
   reads a file into text and then uses the :class:`.TextParser`,
   from which it inherits, to parse the dictionary.

In addition to the  :class:`.TextParser`, a  :class:`.TemplateTextParser` exists.
This parser allows you to use `Jinja2 <http://jinja.pocoo.org/>`_ templates in your
configuration text/files. This allows you to use variables and control structures
that can be filled in at runtime (see also :ref:`cli`).

.. note:: The :class:`.TemplateFileParser` defines a ``__file__`` and ``__filedir__``
    variable pointing to the file that is being parsed and the directory in which
    the file is located, respectively.

.. note:: When using ``Template`` based parsers with the :ref:`cli`, all missing variables
    will be asked for as command line arguments.

Typical structure for SimpleDictParsers
---------------------------------------

Configuration files for :class:`.SimpleDictParser` parsers (and subclasses) are dictionaries that require
the following keys:

1. ``data_loaders`` - a list of dictionaries, each of which defines a :class:`.DataLoader`
   object. A ``name`` key is required for each dictionary to link it to the
   appropriate :class:`.Pipeline` s object.

2. ``pipelines`` - a list of dictionaries, each of which defines a :class:`.Pipeline`
   object. A ``data_from`` key is required that specifies the ``name``
   of the :class:`.DataLoader` object from which the :class:`.Pipeline` should load its data.

3. ``config`` - a dictionary of configuration options used for additional configuration
   of helper classes like the :class:`.Runner` and logging.

.. note:: When the special key ``callable`` is used in a dictionary, the value of that key
          will be treated as a callable object. A :class:`.Finder` will search for all
          :class:`.Callable` objects in the :mod:`.brain_pipe` module and extra paths
          defined in the ``config`` dictionary under the ``extra_paths`` keys. The
          other keys in the dictionary will be passed as keyword arguments to the
          callable object.

.. note:: If you only want to pass a reference to a ``callable``, the ``is_pointer``
          keyword can be used.


Specifying the parser
---------------------

By default, the CLI (see also :ref:`cli`) will try to use the most appropriate
parser for the given input. Currently, the :class:`.YAMLTemplateFileParser` is the
most common default, as it supports `YAML <http://yaml.org/>`_ and `JSON <http://json.org/>`_
files with or without `Jinja2 <http://jinja.pocoo.org/>`_ templates.
