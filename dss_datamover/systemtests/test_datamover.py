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
    "get_system_config_dict",
    "get_pytest_configs",
    "clear_datamover_cache",
    "setup_data_dir",
    "generate_full_ip_prefix",
    "reset_master_obj",
    "setup_large_file",
    "setup_empty_file"
)
class TestDataMover:
    """
    This class may be used for system level / functional tests.
    """

    def test_put_operation(self, get_master, reset_master_obj, setup_data_dir):
        print("Testing PUT operation..")
        get_master.operation = "PUT"
        get_master.start()
        process_put_operation(get_master)
        assert get_master.testcase_passed.value

    # def test_compaction(self, get_master, reset_master_obj):
    #     print("Testing compaction..")
    #     get_master.operation = "PUT"
    #     get_master.start()
    #     get_master.compaction()

    def test_get_operation(self, get_master, reset_master_obj, setup_data_dir):
        print("Testing GET operation..")
        get_master.operation = "GET"
        get_master.start()
        process_get_operation(get_master)
        assert get_master.testcase_passed.value

    def test_list_operation(self, get_master, reset_master_obj, setup_data_dir):
        print("Testing LIST Operation")
        get_master.operation = "LIST"
        get_master.start()
        process_list_operation(get_master)

    def test_delete_operation(self, get_master, reset_master_obj, setup_data_dir):
        print("Testing DEL operation")
        get_master.operation = "DEL"
        get_master.start()
        process_del_operation(get_master)
        assert get_master.testcase_passed.value

    def test_put_large_file(self, get_master, reset_master_obj, setup_data_dir, setup_large_file, get_pytest_configs):
        print("Testing PUT operation with large file")
        get_master.operation = "PUT"
        get_master.fs_config['nfs'] = {'127.0.0.1': get_pytest_configs['large_file_path']}
        get_master.prefix = '127.0.0.1' + get_pytest_configs['large_file_path'] + '/'
        get_master.start()
        assert get_master.testcase_passed.value

    def test_put_empty_file(self, get_master, reset_master_obj, setup_data_dir, setup_empty_file, get_pytest_configs):
        print("Testing PUT operation with empty file")
        get_master.operation = "PUT"
        get_master.fs_config['nfs'] = {'127.0.0.1': get_pytest_configs['empty_file_path']}
        get_master.prefix = '127.0.0.1' + get_pytest_configs['empty_file_path'] + '/'
        get_master.start()
        assert get_master.testcase_passed.value
