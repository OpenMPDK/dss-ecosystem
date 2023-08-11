import os
import sys
import pytest
import re
from queue import Queue
from multiprocessing import Value

from master_application import (
    process_put_operation,
    process_list_operation,
    process_get_operation,
    process_del_operation
)
from utils.config import Config
from conftest import MockNFSCluster, MockOSDirOperations
from task import iterate_dir, indexing


def validate_task_queue(task_queue, nfs_config, prefix=""):
    queue_size = task_queue.qsize()
    nfs_size = 0
    if not prefix:
        for ip_address in nfs_config.keys():
            nfs_size += len(nfs_config[ip_address])
        assert queue_size == nfs_size
        for i in range(queue_size):
            task = task_queue.get()
            assert task.params['nfs_cluster'] in nfs_config
            assert task.params['nfs_share'] in nfs_config[task.params['nfs_cluster']]
    else:
        for i in range(queue_size):
            task = task_queue.get()
            path = task.params['nfs_cluster'] + task.params['nfs_share']
            assert path.startswith(prefix)


@pytest.mark.usefixtures(
    "get_master",
    "shutdown_master",
    "shutdown_master_without_nfscluster",
    "get_system_config_dict",
    "get_pytest_configs",
    "clear_datamover_cache",
    "get_master_for_indexing",
    "get_mock_logger",
    "get_indexing_kwargs"
)
class TestDataMover:
    """
    This class may be used for system level / functional tests. However, it is not confirmed if this is the direction we want to go in,
    hence, these operations are commented out for now
    """

    def test_put_operation(self, clear_datamover_cache, get_master, shutdown_master):
        master = get_master("PUT")
        process_put_operation(master)
        master.compaction()
        assert master.testcase_passed.value
        shutdown_master(master)

    def test_get_operation(self, get_master_dryrun, shutdown_master_without_nfscluster):
        master = get_master_dryrun("GET")
        process_get_operation(master)
        assert master.testcase_passed.value
        shutdown_master_without_nfscluster(master)

    def test_list_operation(self, get_master_dryrun, shutdown_master_without_nfscluster):
        master = get_master_dryrun("LIST")
        process_list_operation(master)
        shutdown_master_without_nfscluster(master)

    def test_delete_operation(self, get_master_dryrun, shutdown_master_without_nfscluster):
        master = get_master_dryrun("DEL")
        process_del_operation(master)
        assert master.testcase_passed.value
        shutdown_master_without_nfscluster(master)

    def test_remove_cache(self, clear_datamover_cache, get_pytest_configs):
        cache_files = get_pytest_configs["cache"]
        cache_exists = False
        for f in cache_files:
            if os.path.exists(f):
                cache_exists = True
        assert not cache_exists, "Failure: DataMover cache not cleared.."

    def test_start_indexing(self, mocker, get_master_for_indexing, get_mock_logger):
        mocker.patch('master_application.NFSCluster', MockNFSCluster)
        master = get_master_for_indexing
        master.logger = get_mock_logger
        # positive no prefix
        master.start_indexing()
        validate_task_queue(master.task_queue, master.nfs_cluster_obj.config)
        # positive with prefix
        master.nfs_cluster_obj.umount_all()
        master.prefix = 'cluster1/mnt/size/'
        master.start_indexing()
        validate_task_queue(master.task_queue, master.nfs_cluster_obj.config, master.prefix)
        # invalid prefix
        master.nfs_cluster_obj.umount_all()
        master.prefixes.clear()
        master.prefix = '/cluster1/mnt/'
        master.start_indexing()
        assert re.match(r'Bad prefix.*', get_mock_logger.get_last('fatal'))
        # mount failure
        master.nfs_cluster_obj.umount_all()
        master.prefixes.clear()
        master.prefix = 'notexist/data/'
        master.start_indexing()
        assert master.indexing_started_flag.value == -1
        assert re.match(r'Mounting failed.*', get_mock_logger.get_last('fatal'))

    def test_iterate_dir(self, mocker):
        mocker.patch('os.scandir', mock_os_scan_dir)
        for result in iterate_dir(data="", logger=None, max_index_size=2, resume_flag=False):
            assert len(result["files"]) <= 2
            assert result["size"] == 200

    def test_indexing(self, mocker, get_indexing_kwargs):
        mocker.patch('os.access', MockOSDirOperations.mock_os_access)
        mocker.patch('task.iterate_dir', MockOSDirOperations.mock_iterate_dir)
        # positive case
        indexing(**get_indexing_kwargs)
        assert get_indexing_kwargs["index_msg_count"].value == 1
        assert get_indexing_kwargs["task_queue"].qsize() == 0
        assert get_indexing_kwargs["index_data_count"].value == 2
        # positive case standalone
        get_indexing_kwargs["standalone"] = True
        indexing(**get_indexing_kwargs)
        assert get_indexing_kwargs["index_msg_count"].value == 1
        assert get_indexing_kwargs["task_queue"].qsize() == 1
        assert get_indexing_kwargs["index_data_count"].value == 4
        # negative case access fail
        mocker.patch('os.access', MockOSDirOperations.mock_os_access_failure)
        indexing(**get_indexing_kwargs)
        assert get_indexing_kwargs["index_msg_count"].value == 1
        assert get_indexing_kwargs["task_queue"].qsize() == 1
        assert get_indexing_kwargs["index_data_count"].value == 4
        assert re.match(r'Read permission.*', get_indexing_kwargs["logger"].get_last("error"))
        # negative case no 'dir'
        mocker.patch('os.access', MockOSDirOperations.mock_os_access)
        mocker.patch('task.iterate_dir', MockOSDirOperations.mock_iterate_dir_no_dir)
        indexing(**get_indexing_kwargs)
        assert get_indexing_kwargs["index_msg_count"].value == 1
        assert get_indexing_kwargs["task_queue"].qsize() == 1
        assert get_indexing_kwargs["index_data_count"].value == 4
        assert re.match(r'.*iterate_dir.*', get_indexing_kwargs["logger"].get_last("error"))
        # positive case no 'files'
        mocker.patch('task.iterate_dir', MockOSDirOperations.mock_iterate_dir_no_files)
        indexing(**get_indexing_kwargs)
        assert get_indexing_kwargs["index_msg_count"].value == 1
        assert get_indexing_kwargs["task_queue"].qsize() == 2
        assert get_indexing_kwargs["index_data_count"].value == 4
        assert get_indexing_kwargs["progress_of_indexing"]["/data"] == 'Pending'
        # negative case no 'size'
        mocker.patch('task.iterate_dir', MockOSDirOperations.mock_oterate_dir_no_size)
        indexing(**get_indexing_kwargs)
        assert get_indexing_kwargs["index_msg_count"].value == 1
        assert get_indexing_kwargs["task_queue"].qsize() == 2
        assert get_indexing_kwargs["index_data_count"].value == 4
