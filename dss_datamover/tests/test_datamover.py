import os
import sys
import pytest
import multiprocessing

from master_application import (
        Master, 
        process_put_operation, 
        process_list_operation, 
        process_get_operation, 
        process_del_operation
    )  # noqa
from utils.config import Config


@pytest.mark.usefixtures("get_master", "shutdown_master", "get_system_config_dict")
class TestDataMover:
    """
    This class may be used for system level / functional tests. However, it is not confirmed if this is the direction we want to go in,
    hence, these operations are commented out for now
    """

    # def test_put_operation(self, get_master, shutdown_master):
    #     master = get_master("PUT")
    #     process_put_operation(master)
    #     assert get_master.testcase_passed.value
    #     shutdown_master(master)

    # def test_get_operation(self, get_master, shutdown_master):
    #     master = get_master("GET")
    #     process_get_operation(master)
    #     assert get_master.testcase_passed.value
    #     shutdown_master(master)
    
    # def test_list_operation(self, get_master, shutdown_master):
    #     master = get_master("LIST")
    #     process_list_operation(master)
    #     assert master.index_data_count.value == 5000
    #     shutdown_master(master)

    # def test_delete_operation(self, get_master, shutdown_master):
    #     master = get_master("DEL")
    #     process_del_operation(master)
    #     assert get_master.testcase_passed.value
    #     shutdown_master(master)
