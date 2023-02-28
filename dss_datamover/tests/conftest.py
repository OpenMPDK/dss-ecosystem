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

from utils import utility, config
from logger import MultiprocessingLogger
from master_application import Master, process_put_operation, process_list_operation, process_get_operation, process_del_operation
from multiprocessing import Queue, Value, Lock

import json
import os
import pytest
import tempfile


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


@pytest.fixture(scope="session")
def get_system_config_object():
    config_obj = config.Config({}, config_filepath="/etc/dss/datamover/config.json")
    return config_obj


@pytest.fixture(scope="session")
def get_system_config_dict(get_system_config_object):
    return get_system_config_object.get_config()


@pytest.fixture
def get_multiprocessing_logger(tmpdir):
    logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
    logger_queue = Queue()
    logger_lock = Lock()
    logging_path = tmpdir
    logging_level = "INFO"

    logger = MultiprocessingLogger(logger_queue, logger_lock, logger_status)
    logger.config(logging_path, __file__, logging_level)
    logger.start()

    yield logger

    # teardown logger
    logger.stop()


@pytest.fixture
def clear_datamover_cache():
    cache_files = ["/var/log/dss/prefix_index_data.json", "/var/log/dss/dm_resume_prefix_dir_keys.txt"]
    for f in cache_files:
        if os.path.exists(f):
            os.remove(f)


class MockSocket():
    """
    Dummy Object for an actual socket, should simulate all basic functions of a socket object
    """
    # TODO: finish building out MockSocket class
    def __init__(self):
        self.mockhost = "192.168.300.144"
        self.mockport = "4000"

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


@pytest.fixture
def get_master_dryrun(get_system_config_dict, get_pytest_configs):
    def instantiate_master_object(operation):
        get_system_config_dict["config"] = get_pytest_configs["config"]
        get_system_config_dict["dest_path"] = get_pytest_configs["dest_path"]
        get_system_config_dict["dryrun"] = True
        master = Master(operation, get_system_config_dict)
        print("instantiated master obj")
        master.start()
        print("successfully started master obj")
        return master
    return instantiate_master_object


@pytest.fixture
def get_master(get_system_config_dict, get_pytest_configs):
    def instantiate_master_object(operation):
        get_system_config_dict["config"] = get_pytest_configs["config"]
        get_system_config_dict["dest_path"] = get_pytest_configs["dest_path"]
        master = Master(operation, get_system_config_dict)
        print("instantiated master obj")
        master.start()
        print("successfully started master obj")
        return master
    return instantiate_master_object


@pytest.fixture
def shutdown_master_without_nfscluster():
    def _method(master):
        print("shutting down master")
        master.stop_logging()
        print("stopping logging")
        master.stop_monitor()
        print("stopping monitoring")
    return _method


@pytest.fixture
def shutdown_master():
    def _method(master):
        print("shutting down master")
        master.nfs_cluster_obj.umount_all()
        print("unmounting nfs cluster")
        master.stop_logging()
        print("stopping logging")
        master.stop_monitor()
        print("stopping monitoring")
    return _method
