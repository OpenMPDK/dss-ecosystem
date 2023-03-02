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

import logging
import os
import prctl
import time
from logging import StreamHandler
from logging.handlers import QueueHandler, RotatingFileHandler
from utils.utility import exception, is_queue_empty
from multiprocessing import Process, Value

"""
Logging Level:
INFO: Verbose mode
DEBUG: Should include DEBUG message
ALERT: Shows minimum required messages
"""

LOGGING_LEVEL = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARN,
    "ERROR": logging.ERROR,
    "EXCEPTION": logging.ERROR,
    "FATAL": logging.FATAL
}

LOGGER_STATE = [
    "NOT-STARTED",
    "RUNNING",
    "STOPPED"
]


class MultiprocessingLogger(object):
    def __init__(self, queue, status, max_file_size=(1024 * 1024), backup_count=5, name='datamover'):
        self.queue = queue
        self.logger_status = status  # 0=NOT-STARTED, 1=RUNNING, 2= STOPPED
        self.stop_logging = Value('i', 0)
        self.app_name = name
        self.process = None
        self.max_log_file_size = max_file_size
        self.log_file_backup_count = backup_count

    @exception
    def get_log_file(self, file_name):
        # print("Log File Name: {}".format(file_name))
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
        if level in LOGGING_LEVEL:
            logging_level = LOGGING_LEVEL[level]
        else:
            logging_level = LOGGING_LEVEL["INFO"]
        return logging_level

    def config(self, path, file_name, logging_level="INFO"):
        self.logging_level = self.set_logging_level(logging_level)
        self.path = path
        self.logfile = self.get_log_file(file_name)

    def create_logger_handle(self):
        '''
        Creates logger handle with Queue. Make sure this is called after starting the logger process.
        Otherwise it ends up with duplicate data
        '''
        try:
            self.logger_handle = logging.getLogger(self.app_name)
            self.logger_handle.addHandler(QueueHandler(self.queue))
            self.logger_handle.setLevel(self.logging_level)
        except Exception as e:
            print('Exception in creating logger handle')
            raise

    def start(self):
        if self.logger_status.value == 0:
            if self.queue:
                self.process = Process(target=self.logging, args=(self.app_name, self.queue, self.logging_level, self.logfile))
                self.process.start()
                self.logger_status.value = 1
            else:
                print("Queue is not present which is required for logger process")
        else:
            self.warn("Logger already started!")

    def stop(self):
        self.info("LOGGER Stopping logging!")
        try:
            self.stop_logging.value = 1
            self.queue.put(None)
            self.process.join()
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

    def create_rotating_file_handler(self, log_file, max_file_size=(1024 * 1024), backup_count=5):
        formatter = logging.Formatter(('%(asctime)s - %(levelname)s - %(message)s'))
        rfh = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
        rfh.setFormatter(formatter)
        return rfh

    @exception
    def logging(self, app_name, queue, log_level, log_file):
        """
        This is the main function which gets executed in a process.
        It consume messages from shared logger_queue and write into log file.
        A filter is applied to print message based on severity.
        :param queue: A shared queue. All the workers produce the messages and logging function consume those message.
        :param lock: A shared lock is used to guard concurrent write operation on shared logger queue.
        :param stop_logging: A atomic shared variable to stop the loop.
        :return: None
        """
        name = "DM_logger"
        prctl.set_name(name)
        prctl.set_proctitle(name)
        stop_flag = False
        stop_counter = 0
        stop_counter_threshold = 5
        logger_handle_thr = None

        try:
            logger_handle_thr = logging.getLogger(app_name)
            logger_handle_thr.setLevel(log_level)
        except Exception as e:
            print('Exception in creating logger handle inside the logger thread')
            raise

        try:
            rfh = self.create_rotating_file_handler(log_file, self.max_log_file_size, self.log_file_backup_count)
            logger_handle_thr.addHandler(rfh)
            logger_handle_thr.addHandler(StreamHandler())

            while True:
                message = queue.get()
                if message is None:
                    stop_flag = True
                else:
                    logger_handle_thr.handle(message)

                if stop_flag:
                    if queue.qsize() and stop_counter < stop_counter_threshold:
                        print("Queue is not empty, but received a stop signal ...")
                        stop_counter += 1
                        time.sleep(1)
                    else:
                        break

        except Exception as e:
            print("Exception in logger process: {}".format(e))

    @exception
    def info(self, message):
        self.logger_handle.info(message)

    @exception
    def debug(self, message):
        self.logger_handle.debug(message)

    @exception
    def warn(self, message):
        self.logger_handle.warning(message)

    @exception
    def error(self, message):
        self.logger_handle.error(message)

    @exception
    def excep(self, message):
        self.logger_handle.exception(message)

    @exception
    def fatal(self, message):
        self.logger_handle.fatal(message)
