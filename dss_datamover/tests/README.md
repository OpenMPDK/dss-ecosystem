# DataMover Pytest Framework

Unit tests for testing DataMover utility and module code. Leverages pytest-cov to generate a code coverage report

## Requirements

This module requires the following modules:

pytest
pytest-mock
pytest-cov
pytest-gitcov

Also refer to requirements.txt file if you would like to install these packages with pip

## Run Pytest

`python3 -m pytest <path to test folder or file> -v -rA --cov=<path to root folder of data mover> --cov-report term --color=yes --disable-warnings`

Here are some examples run from the dss-ecosystem directory


Run all tests by specifying the test folder
`python3 -m pytest dss_datamover/tests -v -rA --cov=dss_datamover/ --cov-report term --color=yes --disable-warnings`

Run on a specific test file
`python3 -m pytest dss_datamover/tests/test_utils.py -v -rA --cov=dss_datamover/ --cov-report term --color=yes --disable-warnings`

Run on a specific test class
`python3 -m pytest dss_datamover/tests/test_utils.py::TestUtils -v -rA --cov=dss_datamover/ --cov-report term --color=yes --disable-warnings`

Run on a specific unittest
`python3 -m pytest dss_datamover/tests/test_utils.py::TestUtils::test_validate_s3_prefix -v -rA --cov=dss_datamover/ --cov-report term --color=yes --disable-warnings`
