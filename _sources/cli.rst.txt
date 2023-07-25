.. _cli:

Command Line Interface
======================

With the command linea interface you can run a configuration file from the command line,
e.g.:

.. code-block:: bash

    $ brain_pipe config_file.extension # e.g. typical_processing.yaml

See also :ref:`configuration` for more information on how to write a configuration file.

API
---
.. code-block::

    usage: [-h] [--parser PARSER] [--parser_files PARSER_FILES]
                       input_file

    Preprocess brain imaging data

    positional arguments:
      input_file            Input file containing the pipeline definitions.

    options:
      -h, --help            show this help message and exit
      --parser PARSER       Name of the parser to use when processing the input
                            file. If not specified, the parser will be determined
                            based on the file extension.
      --parser_files PARSER_FILES, -f PARSER_FILES
                            Path to external file containing the parser to use
                            when processing the input file. Only necessary when
                            defining your own parser. When multiple files are
                            provided, the first parser found will be used. 'None'
                            or '*' can be used to search the files in the
                            brain_pipe package. Default is all files in the
                            brain_pipe package.

