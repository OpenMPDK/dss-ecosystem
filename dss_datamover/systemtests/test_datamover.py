import os
import sys
import pytest

from master_application import (
    process_put_operation,
    process_list_operation,
    process_get_operation,
    process_del_operation
)
from utils.config import Config


@pytest.mark.usefixtures(
    "get_master",
    "shutdown_master",
    "shutdown_master_without_nfscluster",
    "get_system_config_dict",
    "get_pytest_configs",
    "clear_datamover_cache"
)
class TestDataMover:
    """
    This class may be used for system level / functional tests.
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
