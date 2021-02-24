#!/usr/bin/python

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

  def get_s3_client(self,client_name=None):
    # Create S3_Client
    minio_config = self.s3_config.get("minio", {})
    minio_url = minio_config["url"]
    minio_access_key = minio_config["access_key"]
    minio_secret_key = minio_config["secret_key"]
    #start_time = datetime.now()
    s3_client =None
    if self.s3_config.get("client_lib",None).lower() == "minio":
      s3_client = MinioClient(minio_url, minio_access_key, minio_secret_key)
    elif self.s3_config.get("client_lib",None).lower() == "dss_client":
      os.environ["AWS_EC2_METADATA_DISABLED"] = 'true'
      s3_client = DssClientLib(minio_url, minio_access_key, minio_secret_key, self.logger_queue)

    elif self.s3_config.get("client_lib",None).lower() == "boto3":
      config = {"endpoint": "http://202.0.0.103:9000", "minio_access_key": "minio", "minio_secret_key": "minio123"}
      s3_client = S3(config)

    #print("INFO: S3 Connection time: {}".format((datetime.now() - start_time).seconds))
    #self.logger_queue.put("INFO: S3 Connection time: {}".format((datetime.now() - start_time).seconds))
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
        print("EXCEPTION: Unable to terminate Worker-{} - {}".format(self.id, e))
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
          print("EXCEPTION:WORKER-{}:{}".format(self.id, e))
      else:
        #self.logger_queue.put("WWWWWWWWTTTTT ---->>DEBUG: TaskQ Size-{} , empty By worker-{}".format(self.task_queue.qsize(),self.id))
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

      time.sleep(1)  # 1 second delay







