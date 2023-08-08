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
    "clear_datamover_cache"
)
class TestDataMover:
    """
    This class may be used for system level / functional tests.
    """

    def test_put_operation(self, get_master):
        print("Testing PUT operation..")
        get_master.operation = "PUT"
        process_put_operation(get_master)
        get_master.compaction()
        assert get_master.testcase_passed.value

    def test_compaction(self, get_master):
        print("Testing compaction")
        get_master.operation = "PUT"
        get_master.compaction()

    def test_get_operation(self, get_master):
        print("Testing GET operation")
        get_master.operation = "GET"
        process_get_operation(get_master)
        assert get_master.testcase_passed.value

    # def test_list_operation(self, get_master):
    #     print("Testing LIST Operation")
    #     get_master.operation = "LIST"
    #     process_list_operation(get_master)

    def test_delete_operation(self, get_master):
        print("Testing DEL operation")
        get_master.operation = "DEL"
        assert get_master.testcase_passed.value

    def test_remove_cache(self, clear_datamover_cache, get_pytest_configs):
        print("Testing removing DataMover cache")
        cache_files = get_pytest_configs["cache"]
        cache_exists = False
        for f in cache_files:
            if os.path.exists(f):
                cache_exists = True
        assert not cache_exists, "Failure: DataMover cache not cleared.."
