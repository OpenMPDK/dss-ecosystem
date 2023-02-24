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

from utils import utility
from logger import MultiprocessingLogger


@pytest.mark.usefixtures("get_config_dict", "get_multiprocessing_logger")
class TestUtils():

    def test_validate_s3_prefix(self, mocker):
        logger = mocker.patch('logger.MultiprocessingLogger', spec=True).start()
        """ verify validate_s3_prefix function detects faulty prefixes """
        valid_prefix = 'nfsserver01:/mnt/shard/10gb/'
        invalid_prefix = 'nfsserver02:/mnt/shard/15gb'
        assert utility.validate_s3_prefix(logger, valid_prefix)
        assert not utility.validate_s3_prefix(logger, invalid_prefix)
        # TODO: test specific raise exception as well

    def test_loading_config(self, get_config_dict):
        print(f"config dict: {get_config_dict}")
        assert isinstance(get_config_dict, dict)
        assert bool(get_config_dict), "config dict is empty.."

    def test_remote_execution(self, get_config_dict):
        remote_device_configs = get_config_dict["test_remote_device"]
        host, user, passwd = remote_device_configs["host"], remote_device_configs["user"], remote_device_configs["password"]
        remote_command = "ps"
        _, stderr_lines, status = utility.remoteExecution(host, user, passwd, remote_command, blocking=True)
        assert status == 0, f"remote execution failed with status code: {status}, stder: {stderr_lines}"

    def test_first_delimiter_index(self):
        test_string = "usr/lib64/python3/site-packages"
        delimiter = "/"
        assert test_string.index(delimiter) == utility.first_delimiter_index(test_string, delimiter), "failure: inidices mismatched.."
