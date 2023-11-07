"""
The Clear BSD License

Copyright (c) 2023 Samsung Electronics Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Samsung Electronics Co., Ltd. nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import json
import logging
import metrics
import os
import re
import socket
import subprocess
import time
import utils
import uuid


class MinioUSTATCollector(object):
    def __init__(self, configs, seconds, num_iterations,
                 whitelist_patterns, filter=False):
        self.configs = configs
        self.ustat_path = self.configs['ustat_binary_path']
        self.cluster_id = self.configs['cluster_id']
        self.seconds = seconds
        self.num_iterations = num_iterations
        self.whitelist_patterns = whitelist_patterns
        self.filter = filter
        self.TYPE = 'minio'
        self.logger = logging.getLogger('root')

    def poll_statistics(self, metrics_data_buffer, exit_flag):
        minio_uuid = None
        stats_output = {}
        minio_proc_map = {}  # { uuid: proc }

        # spawn a collector and get uuid for each minio instance
        pid_list = self.get_minio_instances()
        collection_time = time.time()
        for pid in pid_list:
            try:
                cmd = (
                    self.ustat_path + ' -p '
                    + str(pid) + ' ' + str(self.seconds)
                    + ' ' + str(self.num_iterations)
                )
                proc_file = os.path.join('/proc', str(pid), 'cmdline')
                proc_cmd = 'minio'
                with open(proc_file) as f:
                    proc_cmd = f.readline()
                minio_uuid = uuid.uuid3(uuid.NAMESPACE_DNS, proc_cmd)
                proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
                stats_output['time'] = collection_time
                minio_proc_map[minio_uuid] = proc
            except Exception as error:
                self.logger.error(f'Error while running USTAT {str(error)}')

        device_subsystem_map = utils.get_device_subsystem_map()
        for minio_uuid, proc in minio_proc_map.items():
            while True:
                if exit_flag.is_set():
                    # shutdown ustat collector
                    self.shutdown_ustat_collector(proc)
                    break

                line = proc.stdout.readline().decode('utf-8')
                if not line or len(line) <= 2 or ('=' not in line):
                    continue

                if line:
                    try:
                        line = line.strip()
                        full_key, value = line.split('=')
                        whitelist_match = False

                        # filter using whitelist patterns if required
                        if self.filter:
                            for regex in self.whitelist_patterns:
                                if re.match(regex, full_key):
                                    whitelist_match = True
                                    break

                        # check if value is a valid promotheus metric value
                        valid_value_flag = value.replace('.', '').isdigit()

                        fields = full_key.split('.')
                        tags = {}
                        tags['cluster_id'] = self.cluster_id
                        tags['target_id'] = socket.gethostname()
                        tags['minio_id'] = str(minio_uuid)
                        tags['type'] = self.TYPE
                        metric_name = fields[-1]

                        if fields[0] == 'nkv':
                            subsystem = fields[2]
                            tags['subsystem_id'] = (
                                device_subsystem_map[subsystem]['nqn']
                            )
                            if 'ip' in metric_name:
                                valid_value_flag = False
                        elif fields[0] == 'minio_upstream':
                            tags['subsystem_id'] = 'minio_upstream'

                        """
                        XOR operation
                        check if filter, then whitelist match should be True
                        if not filter, than whitelist match should be False
                        """
                        if valid_value_flag and self.filter == whitelist_match:
                            metrics_data_buffer.append(
                                metrics.MetricInfo(full_key, metric_name,
                                                   value, tags, time.time())
                            )
                    except Exception as error:
                        self.logger.error('Error handling line %s, Error: %s',
                                          line, str(error))

    def shutdown_ustat_collector(self, proc):
        try:
            proc.terminate()
        except Exception:
            self.logger.warning('failed to terminate ustat', exc_info=True)
            proc.kill()

    def get_minio_instances(self):
        proc_name = re.compile(r"^minio$")
        cmds = ["minio"]
        pid_list = None
        try:
            pid_list = utils.find_process_pid(proc_name, cmds)
        except Exception as error:
            self.logger.error(
                f'Error: unable to get MINIO PID list {str(error)}'
            )
        return pid_list

    def process_stats(self, stats_dict):
        return json.dumps(
            utils.convert_stats_dict_to_json_dict(stats_dict),
            indent=4
        )
