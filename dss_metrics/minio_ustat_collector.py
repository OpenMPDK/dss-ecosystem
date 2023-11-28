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
import re
import socket
import subprocess
import time
import utils

import metrics

from prometheus_client import Metric
from prometheus_client.registry import Collector


class MinioUSTATCollector(Collector):
    def __init__(self, configs, metrics_scopes, seconds, num_iterations,
                 whitelist_patterns, filter=False):
        self.metrics_scopes = metrics_scopes
        self.configs = configs
        self.ustat_path = self.configs['ustat_binary_path']
        self.cluster_id = self.configs['cluster_id']
        self.seconds = seconds
        self.num_iterations = num_iterations
        self.whitelist_patterns = whitelist_patterns
        self.filter = filter
        self.TYPE = 'minio_ustat'
        self.url_prefix = "http://"
        self.cluster_id_url_suffix = "/minio/cluster_id"
        self.logger = logging.getLogger(metrics.APP_NAME)

    def collect(self):
        self.logger.debug("--- MINIO USTAT COLLECTOR ---")

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

                minio_endpoint = utils.get_minio_endpoint_from_process(
                    pid, self.logger)
                minio_uuid = utils.get_minio_cluster_uuid(
                    minio_endpoint, self.url_prefix,
                    self.cluster_id_url_suffix)

                proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
                stats_output['time'] = collection_time
                minio_proc_map[minio_uuid] = proc
            except Exception as error:
                self.logger.error(f'Error while running USTAT {str(error)}')

        device_subsystem_map = utils.get_device_subsystem_map()
        for minio_uuid, proc in minio_proc_map.items():
            lines = proc.stdout.readlines()
            for line in lines:
                line = line.decode('utf-8')
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

                        if valid_value_flag:
                            if (self.filter == whitelist_match) or (
                                                    not self.filter):

                                # check if scope applies to metric
                                scope = self.get_metric_scope(full_key)
                                if scope:
                                    tags['scope'] = scope

                                name = full_key.replace(".", "")
                                metric = Metric(name, full_key, 'gauge')
                                metric.add_sample(
                                    name,
                                    value=value,
                                    labels=tags,
                                    timestamp=time.time()
                                )
                                self.logger.debug(f"collected: {metric}")
                                yield metric
                    except Exception as error:
                        self.logger.error('Error handling line %s, Error: %s',
                                          line, str(error))

    def get_metric_scope(self, key):
        for regex in self.metrics_scopes.keys():
            if re.match(regex, key):
                return self.metrics_scopes[regex]
        return None

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
