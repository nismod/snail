<p align="center">
<a href="https://github.com/nismod/snail/tutorials">Tutorials</a> |
<a href="https://github.com/nismod/snail/issues">Issues</a>
</p>

<p align="center">
<img src="./images/snail.svg" alt="snail" />
</p>

> This code is under early development

# 🤔 What is this?

This is a Python package to help with analysis of the potential impacts of
climate hazards and other perils on infrastructure networks.

## Development

Clone this repository using [GitHub Desktop](https://desktop.github.com/) or on
the command line:

    git clone git@github.com:nismod/snail.git

Change directory into the root of the project:

    cd snail

Run this once, so python will recognise the source code as a package:

    python setup.py develop

Install development packages:

    pip install pytest pytest-cov flake8 nbstripout

Run tests using [pytest](https://docs.pytest.org/en/latest/) and
[pytest-cov](https://pytest-cov.readthedocs.io) to check coverage:

    pytest --cov=snail --cov-report=term-missing tests

Run a linter ([flake](https://flake8.pycqa.org/en/latest/)) to check code
formatting:

    flake8

When working on the tutorial notebooks, it is recommended to install and
configure [nbstripout](https://github.com/kynan/nbstripout) so data and outputs
are not committed in the notebook files:

    nbstripout --install
