#!/usr/bin/python
"""
 *   BSD LICENSE
 *
 *   Copyright (c) 2021 Samsung Electronics Co., Ltd.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Samsung Electronics Co., Ltd. nor the names of
 *       its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import prctl
import time
from utils.utility import exception, is_queue_empty
from multiprocessing import Process, Queue, Value

"""
Logging Level:
INFO: Verbose mode
DEBUG: Should include DEBUG message
ALERT: Shows minimum required messages
"""

LOGGING_LEVEL = [
    "INFO",
    "DEBUG",
    "WARNING",
    "ERROR",
    "EXCEPTION",
    "FATAL"
]

LOGGER_STATE = [
    "NOT-STARTED",
    "RUNNING",
    "STOPPED"
]


class MultiprocessingLogger(object):

    def __init__(self, queue, status):
        self.queue = queue
        self.logger_status = status  # 0=NOT-STARTED, 1=RUNNING, 2= STOPPED
        self.stop_logging = Value('i', 0)
        self.process = None

    @exception
    def get_log_file(self, file_name):
        file_name = file_name.split('/')[-1]
        if not self.path:
            self.path = "/var/log"
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        file_name = file_name.replace("py", "log")
        return os.path.abspath(self.path + "/" + file_name)

    def set_logging_level(self, level):
        """
        Set LOG Level , default is INFO mode
        :param level: INFO|DEBUG|WARN|ERROR|EXCEPTION
        :return: Numeric log level.
        """
        logging_level = 0
        if level in LOGGING_LEVEL:
            logging_level = LOGGING_LEVEL.index(level, 0, 5)

        return logging_level

    def config(self, path, file_name, logging_level="INFO"):
        self.logging_level = self.set_logging_level(logging_level)
        self.path = path
        self.logfile = self.get_log_file(file_name)

    def start(self):
        if self.logger_status.value == 0:
            if self.queue:
                self.process = Process(target=self.logging, args=(self.queue, self.stop_logging))
                self.process.start()
                self.logger_status.value = 1
        else:
            self.warn("Logger already started!")

    def stop(self):
        self.info("Log file is saved at {}".format(self.logfile))
        self.info("LOGGER Stopping logging!")
        while self.process.is_alive():
            self.stop_logging.value = 1
            time.sleep(1)
            if is_queue_empty(self.queue):
                try:
                    self.process.terminate()
                except Exception as e:
                    print("EXCEPTION: LOGGER - {}".format(e))

        print("INFO: LOGGER Stopped !!!")

    def status(self):
        """
        Return the status of logger , 3 state it can be
        NOT-STARTED
        RUNNING
        STOPPED
        :return:
        """
        return LOGGER_STATE[self.logger_status.value]

    @exception
    def logging(self, queue, stop_logging):
        """
        This is the main function which gets executed in a process.
        It consume messages from shared logger_queue and write into log file.
        A filter is applied to print message based on severity.
        :param queue: A shared queue. All the workers produce the messages and logging function consume those message.
        :param stop_logging: A atomic shared variable to stop the loop.
        :return: None
        """
        name = "SDG_logger"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        try:
            print("Log file:{}".format(self.logfile))
            fh = None
            # Move existing log file to log1
            if os.path.exists(self.logfile):
                newfile = self.logfile + ".bak"
                if os.path.exists(newfile):
                    os.remove(newfile)
                os.rename(self.logfile, newfile)
            while True:
                fh = open(self.logfile, "a")

                while not is_queue_empty(queue):
                    message = queue.get()
                    if type(message) == tuple:
                        (message_level, message_value) = message
                        print("{}: {}".format(LOGGING_LEVEL[message_level], message_value))
                        fh.write(str(time.ctime()) + ": " + LOGGING_LEVEL[message_level] + ": " + message_value + "\n")
                    else:
                        fh.write(str(time.ctime()) + ": " + message + "\n")

                if stop_logging.value and queue.qsize() == 0:
                    break
                time.sleep(1)
                fh.close()

        except Exception as e:
            print("Exception: {}".format(e))
        finally:
            if fh:
                fh.close()

    @exception
    def info(self, message):
        msg = (0, message)
        self.queue.put(msg)

    @exception
    def debug(self, message):
        if self.logging_level < len(LOGGING_LEVEL) and LOGGING_LEVEL[self.logging_level] == "DEBUG":
            msg = (1, message)
            self.queue.put(msg)

    @exception
    def warn(self, message):
        if self.logging_level <= 2:
            msg = (2, message)
            self.queue.put(msg)

    @exception
    def error(self, message):
        if self.logging_level <= 3:
            msg = (3, message)
            self.queue.put(msg)

    @exception
    def excep(self, message):
        if self.logging_level <= 4:
            msg = (4, message)
            self.queue.put(msg)

    @exception
    def fatal(self, message):
        if self.logging_level <= 5:
            msg = (5, message)
            self.queue.put(msg)
