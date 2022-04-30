#!/usr/bin/python

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

import os
import sys
from utils.utility import exception, exec_cmd, File
from utils.config import Config, TargetCompactionArgumentParser
from datetime import datetime
import json
import time
import socket

"""
The target_compaction script start the compaction on each target node and wait for its completion.
"""

HOSTNAME = socket.gethostname()


class Compaction:

    def __init__(self, params):
        self.logdir = params["logdir"]
        self.target_ip = params.get("ip_address", None)
        self.user_id = params.get("user_id", None)
        self.password = params.get("password", None)
        self.target_source = params["installation_path"]
        self.logger = self.get_logger()
        self.nqn = self.get_subsystem_nqn(params.get("subsystem_nqn", None))
        self.status = {}
        self.finished_nqn_compaction = 0
        self.compaction_start_time = None

    def __del__(self):
        if self.logger:
            self.logger.close()

    def get_logger(self):
        if not os.path.exists(self.logdir):
            command = "mkdir -p {}".format(self.logdir)
            ret, console = exec_cmd(command, True, True, self.user_id)
            if ret:
                print("ERROR: Unable to create log directory {}".format(self.logdir, console))
        target_log_fh = None
        log_file_name = self.logdir + "/target_compaction.log"
        try:
            target_log_fh = open(log_file_name, "w")
        except Exception as e:
            print("EXCEPTION: {}".format(e))

        return target_log_fh

    def start(self):
        """
        Start the compaction process in a target node when subsystem nqn are specified
        :return: None
        """
        command = "sudo " + self.target_source + "/scripts/dss_rpc.py -s /var/run/spdk.sock rdb_compact -n "
        self.compaction_start_time = datetime.now()
        for nqn in self.nqn:
            compaction_command = command + nqn.strip()
            nqn_compaction_start_time = datetime.now()
            ret, console = exec_cmd(compaction_command, True, True)
            if ret == 0 and console:
                compaction_start_status = json.loads(console)
                if "result" in compaction_start_status:
                    if compaction_start_status["result"] == "STARTED":
                        self.logger.write("INFO: Compaction started for NQN - {}, StartTime-{}\n".format(nqn, nqn_compaction_start_time))
                    else:
                        self.logger.write("INFO: Compaction Status:{} for NQN - {}".format(compaction_start_status["result"], nqn))
                    self.status[nqn] = False
                else:
                    self.logger.write("ERROR: Failed to start compaction for NQN - {}\n {}\n".format(nqn, console))
            else:
                self.logger.write("ERROR: failed to start compaction for NQN - {}".format(nqn))

    def get_subsystem_nqn(self, subsystem_nqn_str):
        subsystem_nqn_list = []
        if subsystem_nqn_str:
            subsystem_nqn_list = subsystem_nqn_str.split(",")
            self.logger.write("INFO: Compaction should be initiated for the following NQNs\n {}\n".format(subsystem_nqn_list))
        else:
            self.logger.write("INFO: Using \"nvme list-subsys\" command for getting subsystem-nqn of {}\n".format(self.target_ip))
            subsystem_nqn_list = self.get_subsystem_nqn_from_command()
        return subsystem_nqn_list

    def get_subsystem_nqn_from_command(self):
        command = 'nvme list-subsys'
        ret, console = exec_cmd(command, True, True)
        nqn = []
        if ret == 0:
            lines = console.split()
            for line in lines:
                line = line.decode('utf-8')
                if line.startswith('NQN'):
                    subsystem_nqn = line.split("=")[-1]
                    fields = subsystem_nqn.split(":")
                    if fields[-1].startswith(HOSTNAME):
                        nqn.append(subsystem_nqn)
        self.logger.write("INFO: Compaction should be initiated for the following NQNs \n {}\n".format(nqn))
        return nqn

    def get_status(self):
        command = "sudo " + self.target_source + "/scripts/dss_rpc.py -s /var/run/spdk.sock rdb_compact --get_status -n "
        for nqn in self.nqn:
            status_command = command + nqn
            ret, console = exec_cmd(status_command, True, True)
            if ret == 0 and console:
                status = json.loads(console)
                if "result" in status and status["result"] == "IDLE":
                    self.status[nqn] = True
                    self.finished_nqn_compaction += 1


if __name__ == "__main__":
    params = TargetCompactionArgumentParser()
    compaction = Compaction(params)
    compaction.start()
    while True:
        compaction.get_status()
        if compaction.finished_nqn_compaction >= len(compaction.status):
            compaction.logger.write("INFO: Compaction is finished! Time-{} Seconds\n".format((datetime.now() - compaction.compaction_start_time).seconds))
            break
        time.sleep(1)
