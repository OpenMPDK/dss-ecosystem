#!/usr/bin/py

"""
# The Clear BSD License
#
# Copyright (c) 2022 Samsung Electronics Co., Ltd.
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

from master_application import Master, process_put_operation, process_list_operation, process_get_operation, process_del_operation
import os
import sys
import unittest
DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(DIR + "/../")
sys.path.append(BASE_DIR)
print(sys.path)


class DataMover(unittest.TestCase):

    config = {'clients_hosts_or_ip_addresses': ['202.0.0.135'],
              'master': {'ip_address': '202.0.0.135', 'workers': 5, 'max_index_size': 500, 'size': '1GB'},
              'client': {'workers': 5, 'max_index_size': 500, 'user_id': 'ansible', 'password': 'ansible'},
              'message': {'port_index': 6000, 'port_status': 6001},
              'nfs_config': {'202.0.0.104': ['/deer', '/dog']},
              's3_storage': {'minio': {'url': '204.0.0.137:9000', 'access_key': 'minio', 'secret_key': 'minio123'},
                             'bucket': 'bucket', 'client_lib': 'minio'},
              'logging': {'path': '/var/log/dss', 'level': 'INFO'},
              'dss_targets': ['202.0.0.104'],
              'environment': {
                  'gcc': {'version': '5.1', 'source': '/usr/local/bin/setenv-for-gcc510.sh', 'required': True}},
              'debug': False, 'dest_path': '/var/log/dss/GET'}

    def test_0_put(self):
        print("################ UnitTest - PUT ################")
        operation = "PUT"

        master = Master(operation, self.config)
        master.start()
        process_put_operation(master)
        self.assertTrue(master.testcase_passed.value)
        master.nfs_cluster_obj.umount_all()
        master.stop_logging()

    def test_1_list(self):
        print("################ UnitTest - LIST ################")
        operation = "LIST"
        master = Master(operation, self.config)
        master.start()
        process_list_operation(master)
        self.assertEqual(master.index_data_count.value, 5000)
        master.stop_logging()

    def test_2_get(self):
        print("################ UnitTest - GET ################")
        operation = "GET"
        master = Master(operation, self.config)
        master.start()
        process_get_operation(master)
        self.assertTrue(master.testcase_passed.value)
        master.stop_logging()

    def test_3_delete(self):
        print("################ UnitTest - DEL ################")
        operation = "DEL"
        master = Master(operation, self.config)
        master.start()
        process_del_operation(master)
        self.assertTrue(master.testcase_passed.value)
        master.stop_logging()

    """
    def test_list_prefix(self):
        #self.master.logger.info("UnitTest - LIST Prefix ")
        #self.result()
        #self.assertTrue(self.master.testcase_passed.value)
    def test_get_prefix(self):
        #self.master.logger.info("UnitTest - GET Prefix ")
        #self.result()
        #self.assertTrue(self.master.testcase_passed.value)
    def test_delete_prefix(self):
        self.master.logger.info("UnitTest - DEL Prefix ")
        self.result()
        #self.assertTrue(self.master.testcase_passed.value)
    """


if __name__ == "__main__":
    unittest.main()
