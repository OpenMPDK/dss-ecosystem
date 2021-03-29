#!/usr/bin/python

"""
 *   BSD LICENSE
 *
 *   Copyright (c) 2019 Samsung Electronics Co., Ltd.
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

import os,sys
import time
import random
import argparse
import platform
from multiprocessing import Process,Queue,Value, Lock
from minio_client import MinioClient
from s3_client import S3
from dss_client import DssClientLib
import time
from task import Task
from datetime import datetime
from utils.utility import exception

"""
TODO:
task_queue shared among the workers, should use same lock by all workers.
logger_queue is shared among the workers and logger 
"""

class Worker:
  def __init__(self,**kwargs):
    self.id = kwargs.get("id", None)
    self.task_queue=kwargs.get("task_queue", None)
    self.index_data_queue = kwargs.get("index_data_queue", None)
    self.logger_queue = kwargs.get("logger_queue", None)  # A multiprocessing logger queue
    self.operation_status_queue = kwargs.get("status_queue", None)  # Used by only client Application

    self.s3_config = kwargs.get("s3_config", None)

    self.status = Value('i', 1)
    self.lock = Lock()
    self.process = None
    #self.index_data_count = kwargs.get("index_data_count", None)

    # Keep track of progress of indexing : Used by Master Application only.
    self.progress_of_indexing = kwargs.get("progress_of_indexing", {} )
    self.progress_of_indexing_lock = kwargs.get("progress_of_indexing_lock", None)

    # Listing variables
    self.listing_progress= kwargs.get("listing_progress", None)
    self.listing_progress_lock = kwargs.get("listing_progress_lock", None)

  def __del__(self):

    self.stop()
    #time.sleep(1)

  def get_s3_client(self):
    """
    Create actual s3_client based on s3 credential
    :return: s3_client connection.
    """
    minio_config = self.s3_config.get("minio", {})
    minio_url = minio_config["url"]
    minio_access_key = minio_config["access_key"]
    minio_secret_key = minio_config["secret_key"]
    s3_client =None

    try:
      s3_client_lib_name = (self.s3_config["client_lib"]).lower()
      if s3_client_lib_name == "minio":
        s3_client = MinioClient(minio_url, minio_access_key, minio_secret_key, self.logger_queue)
      elif s3_client_lib_name == "dss_client":
        os.environ["AWS_EC2_METADATA_DISABLED"] = 'true'
        s3_client = DssClientLib(minio_url, minio_access_key, minio_secret_key, self.logger_queue)

      elif s3_client_lib_name == "boto3":
        config = {"endpoint": "http://202.0.0.103:9000", "minio_access_key": "minio", "minio_secret_key": "minio123"}
        s3_client = S3(config)
      else:
        self.logger_queue.put("ERROR: S3 Client-{} doesn't exist! Supported S3 clients [\"minio\",\"dss_client\", \"boto3\"] ".format(s3_client_lib_name))
    except Exception as e:
      self.logger_queue.put("EXCEPTION: BAD s3_client {}".format(e))

    return s3_client



  def start(self):
    """
    Start the worker process
    :return: None
    """
    try:
      self.process = Process(target=self.run)
      self.process.start()
    except Exception as e:
      self.logger_queue.put("EXCEPTION: {}".format(e))

    self.logger_queue.put("INFO:Worker-{} started ... ".format(self.id))

  def stop(self):
    """
    Stop the worker process
    :return: None
    """
    if self.status.value:
      self.status.value = 0

    # Make sure process is stopped
    if self.process.is_alive() :
      time.sleep(2)
      try:
        self.process.terminate()
      except Exception as e:
        self.logger_queue.put("EXCEPTION: Unable to terminate Worker-{} - {}".format(self.id, e))
    self.logger_queue.put("INFO: Worker-{} stopped!".format(self.id))


  def get_status(self):
    """
    Return the worker status!
    :return:
    """
    return self.status.value

  def run(self):
    """
    Run the actual process
    :return:
    """

    s3_client = self.get_s3_client()

    if not s3_client:
      self.status.value = 0
      return

    while True:
      # Get the status of worker from a shared flag.
      if not self.status.value:
        break
      # Get the task from shared task_queue, shared among the workers.
      task = None

      if self.task_queue and self.task_queue.qsize():
        try:
          #print("DEBUG: TaskQ Size-{}, worker-{}".format(self.task_queue.qsize(), self.id))
          #self.logger_queue.put("DEBUG: TaskQ Size-{}, worker-{}".format(self.task_queue.qsize(),self.id))
          task = self.task_queue.get()
        except Exception as e:
          self.logger_queue.put("EXCEPTION:WORKER-{}:{}".format(self.id, e))
      else:
        pass

      if task:
        #print("DEBUG: Task-{} is being processed by worker-{}".format(task.id, self.id))
        #self.logger_queue.put("DEBUG: Task-{} is being processed by worker-{}".format(task.id, self.id))
        # Start Task - May be a PUT,LIST,DEL,GET or indexing operation
        task.start(task_queue=self.task_queue,
                   index_data_queue=self.index_data_queue,
                   status_queue=self.operation_status_queue,
                   logger_queue=self.logger_queue,
                   progress_of_indexing=self.progress_of_indexing,
                   progress_of_indexing_lock=self.progress_of_indexing_lock,
                   listing_progress = self.listing_progress,
                   listing_progress_lock = self.listing_progress_lock,
                   s3_client = s3_client
                   )

      #time.sleep(1)  # 1 second delay







