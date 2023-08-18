#!/usr/bin/python3

"""
# The Clear BSD License
#
# Copyright (c) 2023 Samsung Electronics Co., Ltd.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Samsung Electronics Co., Ltd. nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from utils import config

import json
import os
import pytest


@pytest.fixture(scope="session")
def get_pytest_configs():
    pytest_config_filepath = os.path.dirname(__file__) + "/pytest_config.json"
    with open(pytest_config_filepath) as f:
        pytest_configs = json.load(f)
    return pytest_configs


@pytest.fixture(scope="session")
def get_config_object():
    test_config_filepath = os.path.dirname(__file__) + "/pytest_config.json"
    config_obj = config.Config({}, config_filepath=test_config_filepath)
    return config_obj


@pytest.fixture(scope="session")
def get_config_dict(get_config_object):
    return get_config_object.get_config()


@pytest.fixture
def clear_datamover_cache(get_pytest_configs):
    cache_files = get_pytest_configs["cache"]
    for f in cache_files:
        if os.path.exists(f):
            os.remove(f)


class MockSocket():
    """
    Dummy Object for an actual socket, should simulate all basic functions of a socket object
    """
    # TODO: finish building out MockSocket class
    def __init__(self):
        self.host = "xxx.xxxx.xxxx.xxxx"
        self.port = "xxxx"

    def connect(self):
        return True

    def recv(self):
        pass

    def sendall(self):
        pass


@pytest.fixture
def get_mock_clientsocket(mocker):
    mock_clientsocket = mocker.patch('socket_communication.ClientSocket', spec=True)
    return mock_clientsocket


@pytest.fixture
def get_mock_serversocket(mocker):
    mock_serversocket = mocker.patch('socket_communication.ClientSocket', spec=True)
    return mock_serversocket
