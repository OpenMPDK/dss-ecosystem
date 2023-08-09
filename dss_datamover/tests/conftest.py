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
from logger import MultiprocessingLogger
from master_application import Master
from multiprocessing import Queue, Value, Lock
from socket_communication import ClientSocket, ServerSocket

import json
import os
import socket
import pytest
from enum import Enum


class Status(Enum):
    NORMAL = 0
    CONNECTIONERROR = 1
    CONNECTIONREFUSEERROR = 2
    SOCKETTIMEOUT = 3
    SOCKETERROR = 4
    EXCEPTION = 5
    MISALIGNEDBUFSIZE = 6
    WRONGBUFSIZE = 7
    LISTENING = 8
    CLOSED = 9


@pytest.fixture(scope="session")
def get_pytest_configs():
    pytest_config_filepath = os.path.dirname(__file__) + "/pytest_config.json"
    with open(pytest_config_filepath) as f:
        pytest_configs = json.load(f)
    return pytest_configs


@pytest.fixture(scope="session")
def get_config_object():
    # test_config_filepath = os.path.dirname(__file__) + "/pytest_config.json"
    test_config_filepath = "/etc/dss/datamover/standard_config.json"
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
    logger_lock = Value('i', 0)
    logging_path = tmpdir
    logging_level = "INFO"

    logger = MultiprocessingLogger(logger_queue, logger_lock, logger_status)
    logger.config(logging_path, __file__, logging_level)
    logger.create_logger_handle()
    logger.start()

    yield logger

    # teardown logger
    logger.stop()


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
    def __init__(self, family=0, type=0, proto=0, fileno=0):
        self.timeout = 0
        self.status = Status.NORMAL
        self.data = ''
        self.data_index = 0  # indicates the starting pos of the sending data when calling recv
        self.max_bufsize = 10  # maximum length of return data when calling recv

    def connect(self, address):
        if self.status == Status.CONNECTIONERROR:
            raise ConnectionError
        elif self.status == Status.CONNECTIONREFUSEERROR:
            raise ConnectionRefusedError
        elif self.status == Status.SOCKETERROR:
            raise socket.error
        elif self.status == Status.SOCKETTIMEOUT:
            raise socket.timeout
        else:
            return

    def recv(self, bufsize):
        if self.status == Status.CONNECTIONERROR:
            raise ConnectionError
        elif self.status == Status.SOCKETTIMEOUT:
            raise socket.timeout
        elif self.status == Status.EXCEPTION:
            raise Exception
        elif self.status == Status.MISALIGNEDBUFSIZE:
            ret = self.data[self.data_index: self.data_index + bufsize + 1]
            return ret
        else:
            ret = ''
            if not self.data:
                return ret
            if self.data_index >= len(self.data):
                raise Exception
            if bufsize > self.max_bufsize:
                bufsize = self.max_bufsize
            if bufsize >= len(self.data) - self.data_index:
                ret = self.data[self.data_index:]
                self.data_index = len(self.data)
            else:
                ret = self.data[self.data_index: self.data_index + bufsize]
                self.data_index += bufsize
            return ret.encode("utf8", "ignore")

    def send(self, data, flags=None):
        return self.sendall(data, flags)

    def sendall(self, data, flags=None):
        self.data = ''
        self.data_index = 0
        if self.status == Status.CONNECTIONERROR:
            raise ConnectionError
        elif self.status == Status.CONNECTIONREFUSEERROR:
            raise ConnectionRefusedError
        elif self.status == Status.SOCKETERROR:
            raise socket.error
        elif self.status == Status.SOCKETTIMEOUT:
            raise socket.timeout
        else:
            self.data = data
        return

    def setsockopt(self, param1, param2, param3):
        pass

    def settimeout(self, new_timeout):
        pass

    def close(self):
        if self.status == Status.LISTENING or self.status == Status.NORMAL:
            self.status = Status.CLOSED
        else:
            raise Exception

    def listen(self, backlog):
        self.status = Status.LISTENING

    def bind(self, address):
        if self.status == Status.NORMAL:
            return
        else:
            raise Exception

    def get_default_ip(self):
        default_ip = ""
        pytest_config_filepath = os.path.dirname(__file__) + "/pytest_config.json"
        with open(pytest_config_filepath) as f:
            pytest_configs = json.load(f)
            default_ip = pytest_configs['default_ip']
        return default_ip

    def accept(self):
        return self, (self.get_default_ip(), 1234)


@pytest.fixture
def get_header_length(mocker, get_config_dict):
    return get_config_dict.get("socket_options", {}).get("response_header_length", 10)


class MockLogger():

    def __init__(self):
        self.logs = {'error': [], 'info': [], 'warn': [], 'excep': [], 'fatal': [], 'debug': []}

    def error(self, msg):
        self.logs['error'].append(msg)

    def info(self, msg):
        self.logs['info'].append(msg)

    def warn(self, msg):
        self.logs['warn'].append(msg)

    def excep(self, msg):
        self.logs['excep'].append(msg)

    def fatal(self, msg):
        self.logs['fatal'].append(msg)

    def debug(self, msg):
        self.logs['debug'].append(msg)

    def get_last(self, type):
        ret = ''
        if len(self.logs[type]) > 0:
            ret = self.logs[type][-1]
        return ret

    def clear(self):
        for key in self.logs:
            self.logs[key].clear()


@pytest.fixture
def get_mock_logger():
    return MockLogger()


@pytest.fixture
def get_mock_clientsocket(mocker):
    mock_clientsocket = mocker.patch('socket_communication.ClientSocket', spec=True)
    return mock_clientsocket


@pytest.fixture
def get_mock_serversocket(mocker):
    mock_serversocket = mocker.patch('socket_communication.ServerSocket', spec=True)
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
def get_master_for_indexing(get_system_config_dict, get_pytest_configs):
    get_system_config_dict["config"] = get_pytest_configs["config"]
    get_system_config_dict["dest_path"] = get_pytest_configs["dest_path"]
    master = Master("PUT", get_system_config_dict)
    return master


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


class MockMinio():
    def __init__(self):
        self.data = {}

    def list(self, key=''):
        if not key:
            return list(self.data.items())
        return self.data[key].items() if isinstance(self.data[key], dict) else [(key, self.data[key])]

    def get(self, key):
        if key in self.data:
            return self.data[key]
        return None

    def put(self, key, value):
        if key in self.data:
            self.data[key] = value
            return True
        return False

    def delete(self, key):
        if key in self.data:
            self.data.pop(key)
            return True
        return False


class MockNFSCluster():
    def __init__(self, config={}, user_id="ansible", password="password", logger=None):
        self.local_mounts = {}
        self.config = {'cluster1': ['/data/pics', '/data/docs', '/mnt/size/5GB', '/mnt/size/10GB'],
                       'cluster2': ['/data/test/put', '/bird', '/mnt/block/4KB', '/mnt/size/10MB']}
        self.mounted = False

    def mount_all(self):
        for ip, nfs_shares in self.config.items():
            self.local_mounts[ip] = nfs_shares
        self.mounted = True

    def umount_all(self):
        self.local_mounts.clear()
        self.mounted = False

    def get_mounts(self):
        return self.local_mounts

    def mount_based_on_prefix(self, prefix):
        cluster_ip = ""
        mounted_nfs_share = ""
        ret = 0
        for ip, nfs_share in self.config.items():
            if prefix.startswith(ip):
                cluster_ip = ip
        if not cluster_ip:
            ret = 1
        else:
            for nfs_share in self.config[cluster_ip]:
                nfs_share_prefix = cluster_ip + nfs_share
                if nfs_share_prefix.startswith(prefix):
                    self.mount(cluster_ip, nfs_share)
                    mounted_nfs_share = nfs_share
                    break
        return cluster_ip, mounted_nfs_share, ret, ""

    def mount(self, cluster_ip, nfs_share):
        self.local_mounts[cluster_ip] = nfs_share
        self.mounted = True


@pytest.fixture
def get_mock_nfs_cluster():
    return MockNFSCluster()


@pytest.fixture
def get_indexing_kwargs():
    return {"data": "", "task_queue": Queue(), "logger": MockLogger(), "index_data_queue": Queue(),
            "nfs_cluster": "", "nfs_share": "", "max_index_size": 2, "indexing_started_flag": Value('i', 0),
            "index_data_count": Value('i', 0), "index_msg_count": Value('i', 0), "index_data_queue_size": 10000, "progress_of_indexing": {},
            "prefix_index_data": {}, "standalone": False, "params": {"s3config": "dss_client", "dryrun": False, "resume_flag": False}}


class MockEntry():

    class EntryStat():
        def __init__(self, size):
            self.st_size = size

    def __init__(self, name, size):
        self.stat_ = self.EntryStat(size)
        self.path = ""
        self.name = name

    def stat(self):
        return self.stat_

    def is_dir(self):
        return False


def mock_os_scan_dir(dir):
    return [
        MockEntry("file0", 0), MockEntry("file1", 100), MockEntry("file2", 100),
        MockEntry("file3", 100), MockEntry("file4", 100)
    ]


def mock_os_access(dir, flag):
    return True


def mock_os_access_failure(dir, flag):
    return False


def mock_iterate_dir(**kwargs):
    file_set = ['file1', 'file2']
    yield {"dir": "/data", "files": file_set, "size": 200}


def mock_iterate_dir_no_dir(**kwargs):
    yield {"files": ['file1'], "size": 200}


def mock_iterate_dir_no_files(**kwargs):
    yield {"dir": "/data", "size": 200}


def mock_oterate_dir_no_size(**kwargs):
    yield {"dir": "/data", "size": 0}
