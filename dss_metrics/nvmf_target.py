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

from prometheus_client import Metric
import pexpect
import socket
import time

import config
import utils


# TODO: process at the top level --> metrics.py
class NVMFTarget():
    def __init__(self, seconds, num_iterations):
        self.ustat_path = config.USTAT_BINARY
        self.nvmf_pid, status = utils.check_spdk_running()
        self.seconds = seconds
        self.num_iterations = num_iterations
        self.TYPE = 'target'

    # TODO: perform whitelisting at this level
    def poll_statistics(self):
        try:
            cmd = (
                self.ustat_path + ' -p '
                + str(self.nvmf_pid) + ' '
                + str(self.seconds) + ' '
                + str(self.num_iterations)
            )
            proc = pexpect.spawn(cmd, timeout=self.seconds + 1)
        except Exception as e:
            print(f'Caught exception while running USTAT {str(e)}')
            return

        subsystem_num_to_nqn_map = {}
        drive_num_to_drive_serial_map = {}
        target_metrics = {}  # {full_key: metric}
        while not proc.eof():
            line = proc.readline().decode('utf-8')
            line = line.strip()
            if line:
                try:
                    full_key, value = line.split('=')

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
                    tags['cluster_id'] = 'c01'
                    tags['target_id'] = socket.gethostname()
                    tags['type'] = self.TYPE
                    if 'subsystem' in subsystem_num:
                        tags['subsystem_id'] = (
                            subsystem_num_to_nqn_map[subsystem_num]
                        )
                    else:
                        continue  # skip if subsystem not mentioned

                    if valid_value_flag:
                        metric = Metric(metric_name, full_key, 'gauge')
                        metric.add_sample(
                            metric_name,
                            value=value,
                            labels=tags,
                            timestamp=time.time()
                        )
                        target_metrics[full_key] = metric
                except Exception as error:
                    print(f'Failed to handle line {line}, Error: {str(error)}')

        try:
            ret = proc.terminate()
        except Exception:
            print('ustat process termination exception ', exc_info=True)

        return target_metrics
