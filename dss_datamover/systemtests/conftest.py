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
from master_application import Master, Client
from nfs_cluster import NFSCluster
from multiprocessing import Queue, Value, Lock

import json
import os
import pytest
from multiprocessing import Queue, Value, Lock, Manager, Event
import shutil


@pytest.fixture(scope="session")
def get_pytest_configs():
    pytest_config_filepath = os.path.dirname(__file__) + "/pytest_config.json"
    with open(pytest_config_filepath) as f:
        pytest_configs = json.load(f)
    return pytest_configs


@pytest.fixture(scope="session")
def clear_datamover_cache(get_pytest_configs):
    print("Clearing DataMover cache..")
    cache_files = get_pytest_configs["cache"]
    for f in cache_files:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture
def setup_data_dir(get_pytest_configs):
    # setup data directory
    data_dir = get_pytest_configs["dest_path"]
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    yield  # execute test case
    # cleanup data directory
    shutil.rmtree(data_dir)


@pytest.fixture(scope="session")
def get_system_config_object():
    config_obj = config.Config({}, config_filepath="/etc/dss/datamover/config.json")
    return config_obj


@pytest.fixture(scope="session")
def get_system_config_dict(get_system_config_object):
    return get_system_config_object.get_config()


@pytest.fixture(scope="session")
def generate_full_ip_prefix(get_system_config_dict, get_pytest_configs):
    # current system test design only supports 1 NFS ip and 1 prefix
    for ip, _ in get_system_config_dict["fs_config"]["nfs"].items():
        nfs_ip = ip
        break

    full_prefix = nfs_ip + get_pytest_configs["prefix"]
    return full_prefix


@pytest.fixture
def get_multiprocessing_logger(tmpdir):
    logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
    logger_queue = Queue()
    logger_lock = Lock()
    logging_path = tmpdir
    logging_level = "INFO"

    logger = MultiprocessingLogger(logger_queue, logger_status)
    logger.logging_level = logging_level
    logger.create_logger_handle()
    logger.config(logging_path, __file__, logging_level)
    logger.start()

    yield logger

    # teardown logger
    logger.stop()


@pytest.fixture(scope="session")
def get_master(clear_datamover_cache, get_system_config_dict, get_pytest_configs, generate_full_ip_prefix):
    print("Setting up Master Object..")

    def instantiate_master_object():
        get_system_config_dict["config"] = get_pytest_configs["config"]
        get_system_config_dict["dest_path"] = get_pytest_configs["dest_path"]
        get_system_config_dict["prefix"] = generate_full_ip_prefix
        master = Master("PUT", get_system_config_dict)
        print("instantiated master obj")
        return master
    master = instantiate_master_object()
    yield master
    print("shutting down master")

    if hasattr(master, 'nfs_cluster_obj'):
        print("unmounting nfs cluster")
        master.nfs_cluster_obj.umount_all()

    if master.logger_status.value == 1:
        print("stopping logging")
        master.stop_logging()

    if hasattr(master, 'monitor'):
        print("stopping monitoring")
        master.stop_monitor()


@pytest.fixture
def reset_master_obj(get_master):
    yield  # handoff back to testcase
    print("resetting master object..")
    get_master.stop_workers()
    get_master.stop_clients(force_flag=False)
    if hasattr(get_master, 'monitor'):
        print("stopping monitoring")
        get_master.stop_monitor()

    # teardown/ reset master object status to be used for next testcase
    obj = get_master
    manager = Manager()

    obj.workers = []
    obj.clients = []
    obj.task_queue = Queue()
    obj.task_lock = Lock()
    obj.process_monitor_event = Event()
    obj.lock = Lock()

    # Operation PUT/GET/DEL/LIST
    obj.index_data_queue = Queue()  # Index data stored by workers
    obj.index_data_lock = Lock()
    obj.index_data_generation_complete = Value('i', 0)  # TODO - Set value when all indexing is done.
    obj.indexing_started_flag = Value('i', 0)  # [0,1,2,-1] => ["READY", "STARTED", "COMPLETED", "FAILED"]

    # Operation LIST
    obj.prefix = obj.config.get("prefix", None)
    obj.prefixes = []  # TODO need to take multiple prefix from command line and store into list.
    obj.listing_progress = Value('i', 0)
    obj.listing_status = Value('i', 0)  # [0,1,2] => ['NOT STARTED', 'STARTED', 'COMPLETED']
    obj.listing_only = Value('b', False)
    obj.listing_aggregation_status = Value('i', 0)
    obj.listing_objectkey_queue = Queue()

    # Status Progress
    obj.index_data_count = Value('i', 0)  # File Index count shared between worker process.
    obj.index_msg_count = Value('i', 0)  # How many messages have been produced by the producers.
    obj.received_index_msg_count = Value('i', 0)  # How many messages have been consumed by the consumers.
    obj.operation_start_time = None
    obj.operation_end_time = None

    # Keep track of progress of hierarchical indexing.
    obj.progress_of_indexing = manager.dict()
    obj.progress_of_indexing_lock = manager.Lock()

    # Hierarchical indexing of leaf directory, to be used for listing.
    obj.indexing_key = manager.dict()
    obj.indexing_key = manager.Lock()

    # NFS shares
    obj.nfs_shares = []

    # Unit TestCase
    obj.testcase_passed = Value('b', False)

    # Status queue
    if obj.standalone:
        obj.operation_status_queue = Queue()
    else:
        obj.operation_status_queue = None

    obj.prefix_index_data = manager.dict()

    # Get the directory prefix keys that are yet to be resumed for PUT operation
    obj.dir_prefixes_to_resume = list()
    obj.resume_flag = False


@pytest.fixture
def setup_large_file(get_pytest_configs):
    path = get_pytest_configs["large_file_path"]
    name = "large.file"
    if not os.path.exists(path):
        os.mkdir(path)
    file = path + "/" + name
    count = get_pytest_configs["large_file_size"] * 1024 * 1024 // 4
    os.system(f"dd if=/dev/zero of={file} bs=4k count={count}")
    yield
    shutil.rmtree(path)


@pytest.fixture
def setup_empty_file(get_pytest_configs):
    path = get_pytest_configs["empty_file_path"]
    name = "empty.file"
    if not os.path.exists(path):
        os.mkdir(path)
    file = open(path + "/" + name, 'w')
    file.close()
    yield
    shutil.rmtree(path)


@pytest.fixture
def get_nfs_cluster(get_system_config_dict, get_multiprocessing_logger):
    get_system_config_dict["nfs"] = get_system_config_dict["fs_config"]["nfs"]
    get_system_config_dict["server_as_prefix"] = get_system_config_dict["fs_config"]["server_as_prefix"]
    nfs_cluster = NFSCluster(get_system_config_dict, "ansible", "ansible", get_multiprocessing_logger)
    yield nfs_cluster
    nfs_cluster.umount_all()


@pytest.fixture
def get_client(get_system_config_dict, get_multiprocessing_logger):
    client_ip = get_system_config_dict["clients_hosts_or_ip_addresses"][0]
    client = Client(0, client_ip, "PUT", get_multiprocessing_logger, get_system_config_dict, get_system_config_dict["fs_config"]["server_as_prefix"])
    yield client
    client.stop(force_flag=True)
