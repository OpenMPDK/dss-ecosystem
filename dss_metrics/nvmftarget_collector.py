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

import re
import subprocess
import socket
import time

import metrics
import utils


class NVMFTargetCollector():
    def __init__(
        self,
        ustat_binary_path,
        seconds,
        num_iterations,
        whitelist_patterns,
        filter=False
    ):
        self.ustat_path = ustat_binary_path
        self.nvmf_pid, status = utils.check_spdk_running()
        self.seconds = seconds
        self.num_iterations = num_iterations
        self.whitelist_patterns = whitelist_patterns
        self.filter = filter
        self.TYPE = 'target'

    def poll_statistics(self, metrics_data_buffer):
        try:
            cmd = (
                self.ustat_path + ' -p '
                + str(self.nvmf_pid) + ' '
                + str(self.seconds) + ' '
                + str(self.num_iterations)
            )
            proc = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE)
        except Exception as e:
            print(f'Caught exception while running USTAT {str(e)}')
            return

        subsystem_num_to_nqn_map = {}
        drive_num_to_drive_serial_map = {}

        while True:
            line = proc.stdout.readline().decode('utf-8')
            if not line:
                break
            line = line.strip()
            if line:
                try:
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
                    tags['cluster_id'] = 'c01'
                    tags['target_id'] = socket.gethostname()
                    tags['type'] = self.TYPE
                    if 'subsystem' in subsystem_num:
                        tags['subsystem_id'] = (
                            subsystem_num_to_nqn_map[subsystem_num]
                        )
                    else:
                        continue  # skip if subsystem not mentioned

                    # TODO: return tuple with metric info instead of populating
                    """
                    XOR operation
                    check if filter, then whitelist match should be True
                    if not filter, than whitelist match should be False
                    """
                    if valid_value_flag and (self.filter == whitelist_match):
                        metrics_data_buffer.append(
                            metrics.MetricInfo(
                                full_key,
                                metric_name,
                                value,
                                tags,
                                time.time()
                            )
                        )
                except Exception as error:
                    print(f'Failed to handle line {line}, Error: {str(error)}')
        try:
            proc.terminate()
        except Exception:
            print('ustat process termination exception ', exc_info=True)
            proc.kill()
