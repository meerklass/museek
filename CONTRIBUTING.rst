============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given. 

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~
Implement Features
~~~~~~~~~~~~~~~~~~
Write Documentation
~~~~~~~~~~~~~~~~~~~

Submit Feedback
~~~~~~~~~~~~~~~

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Development Setup
------------------

To set up a development environment with code quality tools:

1. Install the package with dev dependencies::

    pip install -e .[dev]

2. Install pre-commit hooks (recommended)::

    pre-commit install

This will automatically format and lint your code before each commit.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. The docs should be updated or extended accordingly. Add any new plugins to the list in README.rst.
3. The pull request should work for Python 3.10.
4. Make sure that the tests pass.
5. Code must be formatted with ``ruff format`` and pass ``ruff check``.
