# DataMover Pytest Framework

Unit tests for testing DataMover utility and module code. Leverages pytest-cov to generate a code coverage report

## Requirements

This module requires the following modules:

pytest
pytest-mock
pytest-cov
pytest-gitcov

Also refer to requirements.txt file if you would like to install these packages with pip

In order to run system level test, such as those found in test_datamover.py you must make sure the correct environment is sourced with access to the compiler. For example;
`source /usr/local/bin/setenv-for-gcc510.sh`

## Run Pytest
You must be in the dss_datamover directory

Structure:
`python3 -m pytest <path to test folder or file> -v -rA --cov=<path to root folder of data mover> --cov-report term --color=yes --disable-warnings`

Here are some examples run from the dss-ecosystem directory


Run all tests by specifying the test folder
`python3 -m pytest tests -v -rA --cov=. --cov-report term --color=yes --disable-warnings`

Run on a specific test file
`python3 -m pytest tests/test_utils.py -v -rA --cov=. --cov-report term --color=yes --disable-warnings`

Run on a specific test class
`python3 -m pytest tests/test_utils.py::TestUtils -v -rA --cov=. --cov-report term --color=yes --disable-warnings`

Run on a specific unittest
`python3 -m pytest tests/test_utils.py::TestUtils::test_validate_s3_prefix -v -rA --cov=. --cov-report term --color=yes --disable-warnings`
