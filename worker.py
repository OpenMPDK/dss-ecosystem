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
from multiprocessing import Process, Value, Queue
from minio_client import MinioClient
from s3_client import S3
from logger import MultiprocessingLogger
import time

WORKER_OPERATION_STATUS = {
    "PUT": ["NFS_READ", "S3_UPLOAD"],
    "LIST": ["S3_LIST","STATUS_SEND"],
    "GET": ["S3_READ", "STATUS_SEND"],
    "DEL": ["S3_DELETE", "STATUS_SEND"],
    "TEST": ["NFS_READ", "S3_UPLOAD", "S3_READ", "MD5SUM_COMPARE", "STATUS_SEND", "DELETE_TEMP_FILES"]
}
class Worker(object):
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", None)
        self.task_queue = kwargs.get("task_queue", None)
        self.index_data_queue = kwargs.get("index_data_queue", None)
        self.logger = kwargs.get("logger", None)  # A multiprocessing logger queue
        self.operation_status_queue = kwargs.get("status_queue", None)  # Used by only client Application
        self.task_count = Value('i', 0)
        self.task_count_previous = 0
        self.latest_task_processed = Queue() # A queue only store the message processed through Task.

        self.s3_client = None
        self.s3_config = kwargs.get("s3_config", None)
        self.aws_log_debug_val = kwargs.get("aws_log_debug_val", None)

        self.status = Value('i', 0)  # [0,1,2] => ["READY","RUNNING","HUNG"]
        self.operation_progress_status_counter = Value('i', 0)  #
        self.operation_progress_counter_previous_value = 0
        self.process = None
        self.index_data_count = kwargs.get("index_data_count", 0)
        self.index_msg_count = kwargs.get("index_msg_count", 0)

        # Keep track of progress of indexing : Used by Master Application only.
        self.progress_of_indexing = kwargs.get("progress_of_indexing", {})
        self.progress_of_indexing_lock = kwargs.get("progress_of_indexing_lock", None)
        self.indexing_started_flag = kwargs.get('indexing_started_flag')

        # Listing variables
        self.listing_progress = kwargs.get("listing_progress", None)
        self.listing_progress_lock = kwargs.get("listing_progress_lock", None)
        self.listing_status = kwargs.get("listing_status", None)
        self.listing_only = kwargs.get("listing_only", None)
        self.listing_objectkey_queue = kwargs.get("listing_objectkey_queue", None)

        # Testing
        self.skip_upload =  kwargs.get("skip_upload", False)


    def __del__(self):
        self.stop()
        # time.sleep(1)

    def create_s3_client(self):
        """
        Create actual s3_client based on s3 credential
        :return: s3_client connection.
        """
        minio_config = self.s3_config.get("minio", {})
        minio_url = minio_config["url"]
        minio_access_key = minio_config["access_key"]
        minio_secret_key = minio_config["secret_key"]

        try:
            s3_client_lib_name = (self.s3_config["client_lib"]).lower()
            if s3_client_lib_name == "minio":
                self.s3_client = MinioClient(minio_url, minio_access_key, minio_secret_key, self.logger)
            elif s3_client_lib_name == "dss_client":
                os.environ["AWS_EC2_METADATA_DISABLED"] = 'true'
                # To enable DSS CLIENT LOGS, uncomment the below 2 lines
                if self.aws_log_debug_val:
                    os.environ['DSS_AWS_LOG'] = str(self.aws_log_debug_val)
                    os.environ['DSS_AWS_LOG_FILENAME'] = 'aws_sdk_' + str(os.getpid()) + '_'
                from dss_client import DssClientLib
                self.logger.debug('PROCESS ENVIRONMENT DETAILS - {}, PID - {}'.format(os.environ, os.getpid()))
                self.s3_client = DssClientLib(minio_url, minio_access_key, minio_secret_key, self.logger)
            elif s3_client_lib_name == "boto3":
                config = {"endpoint": "http://202.0.0.103:9000", "minio_access_key": "minio",
                          "minio_secret_key": "minio123"}
                self.s3_client = S3(config)
            else:
                self.logger.error(
                    "S3 Client-{} doesn't exist! Supported S3 clients [\"minio\",\"dss_client\", \"boto3\"] ".format(
                        s3_client_lib_name))
        except Exception as e:
            self.logger.excep("BAD s3_client {}".format(e))

    def get_s3_client(self):
        """
        Return s3_client 
        """
        return self.s3_client

    def start(self):
        """
        Start the worker process
        :return: None
        """
        try:
            self.create_s3_client()
            if not self.s3_client.status:
              self.status.value = 0
              self.logger.error("S3 Client is not initialized. Exit worker-{}".format(self.id))
              return

            self.process = Process(target=self.run, args=(self.s3_client, ))
            self.process.start()
        except Exception as e:
            self.logger.excep("{}".format(e))
            return

        while self.status.value == 0:
          time.sleep(0.1)
        self.logger.info("Worker-{} started ... ".format(self.id))

    def stop(self):
        """
        Stop the worker process
        :return: None
        """
        if self.status.value:
            self.status.value = 0

        # Make sure process is stopped
        if self.process and self.process.is_alive():
            time.sleep(2)
            try:
                self.process.terminate()
            except Exception as e:
                self.logger.excep("Unable to terminate Worker-{} - {}".format(self.id, e))
        self.logger.info("Worker-{} stopped!".format(self.id))

    def get_status(self):
        """
        Return the worker status!
        :return:
        """
        return self.status.value

    def is_hung(self, operation):
        try:
            # self.logger.info("worker-{}=> {}:{}".format(self.id, self.operation_progress_counter_previous_value, self.operation_progress_status_counter.value))
            # self.logger.info("worker-{}=> {}:{}".format(self.id, self.task_count_previous, self.task_count.value))
            # Detect hung condition
            if self.task_count.value > 0 and self.task_count.value == self.task_count_previous:
                if ( self.operation_progress_status_counter.value > 0 )  and \
                    self.operation_progress_counter_previous_value == self.operation_progress_status_counter.value:
                    hung_state_index = self.operation_progress_counter_previous_value % len(WORKER_OPERATION_STATUS[operation])
                    hung_state_name = WORKER_OPERATION_STATUS[operation][hung_state_index]
                    #hunged_task = self.latest_task_processed.get()
                    #self.logger.info("Task_id:{}".format(hunged_task.id))
                    latest_message = self.latest_task_processed.get()
                    file_index =  int(self.operation_progress_status_counter.value / len(WORKER_OPERATION_STATUS[operation]))
                    # self.logger.info("Index-{}, Total File-{}, FileIndex-{}".format(self.operation_progress_status_counter.value, len(latest_message["files"]), file_index))
                    self.logger.error("Worker-{} Possibly is in Hung state \"{}\", {}/{}".format(self.id,
                                                                                    hung_state_name,
                                                                                    latest_message["dir"],
                                                                                    latest_message["files"][file_index]
                                                                                    ))
                    #self.task_queue.put(hunged_task) # Put the task again.

                    return True
                else:
                    self.operation_progress_counter_previous_value = self.operation_progress_status_counter.value
            else:
                self.task_count_previous = self.task_count.value
                self.operation_progress_counter_previous_value = self.operation_progress_status_counter.value
        except Exception as e:
            self.logger.excep("Hung Detection - {}".format(e))
        return False



    def run(self, s3_client):
        """
        Run the actual process
        :return:
        """
        self.status.value = 1
        while True:
            # Get the status of worker from a shared flag.
            if not self.status.value:
                break
            # Get the task from shared task_queue, shared among the workers.
            if self.task_queue and self.task_queue.qsize() > 0 :
                try:
                    task = self.task_queue.get()
                    if self.latest_task_processed.qsize() > 0:
                        self.latest_task_processed.get() # Empty queue
                    self.latest_task_processed.put(task.params["data"]) # Add latest message
                    self.operation_progress_status_counter.value = 0 # Reset counter

                    with self.task_count.get_lock():
                        self.task_count.value +=1
                    task.start(worker_id=self.id,
                               task_queue=self.task_queue,
                               index_data_queue=self.index_data_queue,
                               status_queue=self.operation_status_queue,
                               logger=self.logger,
                               progress_of_indexing=self.progress_of_indexing,
                               progress_of_indexing_lock=self.progress_of_indexing_lock,
                               index_data_count=self.index_data_count,
                               index_msg_count=self.index_msg_count,
                               listing_progress=self.listing_progress,
                               listing_progress_lock=self.listing_progress_lock,
                               s3_client=s3_client,
                               indexing_started_flag=self.indexing_started_flag,
                               listing_status=self.listing_status,
                               listing_only=self.listing_only,
                               listing_objectkey_queue=self.listing_objectkey_queue,
                               skip_upload=self.skip_upload,
                               operation_progress_status_counter=self.operation_progress_status_counter
                               )
                except Exception as e:
                    self.logger.excep("WORKER-{}:{}".format(self.id, e))
