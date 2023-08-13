
Brain Pipe
==========

[![PyPI](https://img.shields.io/pypi/v/brain_pipe)](https://pypi.org/project/brain-pipe/)
![Python versions](https://img.shields.io/badge/Python%20version-3.8%2C%203.9%2C%203.10%2C%203.11-orange)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/exporl/brain_pipe/LICENSE)
[![Black code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
[![flake8 code style](https://img.shields.io/badge/code%20style-flake8-blue.svg)](https://flake8.pycqa.org/en/latest/)
[![Docs](https://img.shields.io/badge/docs-https%3A%2F%2Fexporl.github.io%2Fbrain_pipe%2F-green)](https://exporl.github.io/brain_pipe/)
![Tests](https://github.com/exporl/brain_pipe/actions/workflows/ci.yaml/badge.svg)
[![Repository](https://img.shields.io/badge/Repository-https%3A%2F%2Fgithub.com%2Fexporl%2Fbrain__pipe-purple)](https://github.com/exporl/brain_pipe)


![Brain Pipe:](docs/source/_images/brain_pipe_github.svg)

Preprocess brain imaging datasets in a fast and re-usable way.

Motivation
-----------

This repository contains code to efficiently preprocess brain imaging datasets in
python3, predominantly for machine learning downstream tasks.

The initial goal of this code was to preprocess the public EEG dataset of the
[ICASSP 2023 Auditory EEG challenge](https://exporl.github.io/auditory-eeg-challenge-2023/) called
[SparrKULee](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND).

Parts of this code were translated from our internal matlab codebase at ExpORL,
maintained primarily by
[Jonas Vanthornhout](https://gbiomed.kuleuven.be/english/research/50000666/50000672/people/members/00077061)
and [Marlies Gillis](https://gbiomed.kuleuven.be/english/research/50000666/50000672/people/members/00123908)

Installation
------------

You can install this repository as a [pip](https://pip.pypa.io/en/stable/) package.

```bash
pip install brain_pipe
```

How to use
----------

You can write your own preprocessing script, see [the docs](https://exporl.github.io/brain_pipe/pipeline.html#small-example),
or you can use the [cli](https://exporl.github.io/brain_pipe/cli.html) with
[configuration files](https://exporl.github.io/brain_pipe/configuration.html) to preprocess your data.

```bash
brain_pipe config_file.extension
# e.g. brain_pipe sparrKULee.yaml
```

Requirements
------------

Python >= 3.8

Contributing
------------

Contributions are welcome. Please open an issue or a pull request.
For more information, see [CONTRIBUTING.md](https://github.com/exporl/brain_pipe/CONTRIBUTING.md).
This package is created and maintained by [ExpORL, KU Leuven, Belgium](https://gbiomed.kuleuven.be/english/research/50000666/50000672).

Example usage and starting guide
--------------------------------

Read the [docs](public/index.html) for a more detailed explanation of the pipeline.

For a simple example, see [the docs](https://exporl.github.io/brain_pipe/pipeline.html##small-example).
For a more elaborate example, see [examples/exporl/sparrKULee.py](https://github.com/exporl/brain_pipe/examples/exporl/sparrKULee.py)
