#!/usr/bin/python

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

import multiprocessing as mp
import sys
import signal
from utils.utility import exception

SIGNAL = {
    1: "SIGHUP",  # Action terminate process
    2: "SIGINT",  # Action terminate process
    6: "SIGABRT",
    9: "SIGKILL",  # Terminate process
    15: "SIGTERM"  # Action termination process
}


class SignalHandler(object):
    def __init__(self):
        self.registered_functions = []

    def add_fn(self, func):
        self.registered_functions.append(func)

    def initiate(self):
        signal.signal(signal.SIGINT, self.handler)
        # signal.signal(signal.SIGABRT, self.handler)
        # signal.signal(signal.SIGBUS, self.handler)
        # signal.signal(signal.SIGKILL, self.handler)
        # signal.signal(signal.SIGTERM, self.handler)

    @exception
    def handler(self, signal, frame):

        if signal in SIGNAL:
            print("INFO: Received {} Signal ... ".format(SIGNAL[signal]))
        else:
            print("INFO: Received {} Signal ... ".format(signal))

        # if mp.current_process().name != 'MainProcess':
        #     return

        for func in self.registered_functions:
            func()
        print("INFO: All functions are done, exiting ...")
        sys.exit()
