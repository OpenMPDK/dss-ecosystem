
import os
import sys
from multiprocessing import Queue, Value, Lock
import pytest

top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(top_dir)

from utils import utility, config
from logger import MultiprocessingLogger
from master_application import Master, process_put_operation, process_list_operation, process_get_operation, process_del_operation  # noqa

@pytest.fixture(scope="session")
def get_config_object():
    test_config_filepath = os.path.dirname(__file__) + "/test_config.json"
    config_obj = config.Config({}, config_filepath=test_config_filepath)
    return config_obj


@pytest.fixture(scope="session")
def get_config_dict(get_config_object):
    return get_config_object.get_config()


@pytest.fixture(scope="session")
def get_system_config_object():
    config_obj = config.Config({})
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

# TODO: add utility to clear the cache, clear_datamover_cache
@pytest.fixture
def clear_datamover_cache():
    cache_files = ["/var/log/dss/prefix_index_data.json", "/var/log/dss/dm_resume_prefix_dir_keys.txt"]
    for f in cache_files:
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
def get_master(get_system_config_dict):
    def instantiate_master_object(operation):
        master = Master(operation, get_system_config_dict)
        print("instantiated master obj")
        master.start()
        print("successfully started master obj")
        return master
    return instantiate_master_object


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
