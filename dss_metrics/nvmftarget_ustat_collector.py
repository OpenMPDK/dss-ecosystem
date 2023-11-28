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

import logging
import re
import socket
import subprocess
import time
import utils

import metrics

from prometheus_client import Metric
from prometheus_client.registry import Collector


class NVMFTargetUSTATCollector(Collector):
    def __init__(self, configs, metrics_scopes, seconds, num_iterations,
                 whitelist_patterns, filter=False):
        self.configs = configs
        self.metrics_scopes = metrics_scopes
        self.ustat_path = self.configs['ustat_binary_path']
        self.cluster_id = self.configs['cluster_id']
        self.nvmf_pid, status = utils.check_spdk_running()
        self.seconds = seconds
        self.num_iterations = num_iterations
        self.whitelist_patterns = whitelist_patterns
        self.filter = filter
        self.TYPE = 'target_ustat'
        self.logger = logging.getLogger(metrics.APP_NAME)

    def collect(self):
        self.logger.debug("--- NVMF Target USTAT COLLECTOR ---")

        try:
            cmd = (
                self.ustat_path + ' -p '
                + str(self.nvmf_pid) + ' '
                + str(self.seconds) + ' '
                + str(self.num_iterations)
            )
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
        except Exception as e:
            self.logger.error(f'Error while running USTAT {str(e)}')
            return

        subsystem_num_to_nqn_map = {}
        drive_num_to_drive_serial_map = {}
        raw_data_queue = []

        while True:
            # loop until no more output left in USTAT
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.decode('utf-8')
                if len(line) <= 2 or ('=' not in line):
                    continue
            except Exception as e:
                self.logger.error(f"error processing Target USTAT: {str(e)}")

            # process nvmf target USTAT line
            try:
                line = line.strip("\n").replace(" ", "")
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
                subsystem_num = fields[1]
                component = fields[2]
                metric_name = fields[-1]

                if component == 'id' and 'nqn' in metric_name:
                    subsystem_num_to_nqn_map[subsystem_num] = value
                elif 'drive' in component and 'serial' in metric_name:
                    drive_num_to_drive_serial_map[component] = value
                elif 'ip' in metric_name:
                    valid_value_flag = False

                tags = {}
                tags['cluster_id'] = self.cluster_id
                tags['target_id'] = socket.gethostname()
                tags['type'] = self.TYPE

                """
                raw data needs to be stored first and then
                populate metrics objects when we are sure
                we have processed the entire ustat output
                """

                data = {
                    "whitelist_match": whitelist_match,
                    "valid_value_flag": valid_value_flag,
                    "subsystem_num": subsystem_num,
                    "full_key": full_key,
                    "metric_name": metric_name,
                    "value": value,
                    "tags": tags,
                    "time": time.time()
                }
                raw_data_queue.append(data)
            except Exception as e:
                self.logger.error(f'Failed to handle {line}, Error: {str(e)}')

        # shutdown USTAT collector subprocess
        self.shutdown_ustat_collector(proc)

        # All of USTAT output processed, now Metric objects can be populated
        for data in raw_data_queue:
            if (
                data['valid_value_flag'] and
                self.filter == data['whitelist_match']
            ):
                if (
                    'subsystem' in data['subsystem_num'] and
                    data['subsystem_num'] in subsystem_num_to_nqn_map
                ):
                    data['tags']['subsystem_id'] = (
                        subsystem_num_to_nqn_map[data['subsystem_num']]
                    )

                # check if scope applies to metric
                scope = self.get_metric_scope(data['full_key'])
                if scope:
                    data['tags']['scope'] = scope

                metric = Metric(data['metric_name'], data['full_key'], 'gauge')
                metric.add_sample(
                    data['metric_name'],
                    value=data['value'],
                    labels=data['tags'],
                    timestamp=data['time']
                )
                self.logger.debug(f"collected: {metric}")
                yield metric

    def shutdown_ustat_collector(self, proc):
        try:
            proc.terminate()
        except Exception:
            self.logger.warning('failed to terminate ustat', exc_info=True)
            proc.kill()

    def get_metric_scope(self, key):
        for regex in self.metrics_scopes.keys():
            if re.match(regex, key):
                return self.metrics_scopes[regex]
        return None
