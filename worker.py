#!/usr/bin/python

import os,sys
import time
import random
import argparse
import platform
from multiprocessing import Process,Queue,Value, Lock
import time
from task import Task

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
    time.sleep(10)

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
    self.lock.acquire()
    if self.status.value:
      self.status.value = 0
    self.lock.release()

    # Make sure process is stopped
    if self.process.is_alive() :
      time.sleep(5)
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
    self.lock.acquire()
    status =  self.status.value
    self.lock.release()
    return status

  def run(self):
    """
    Run the actual process
    :return:
    """

    while True:
      # Get the status of worker from a shared flag.
      self.lock.acquire()
      current_status = self.status.value
      self.lock.release()

      if not current_status:
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
                   listing_progress_lock = self.listing_progress_lock
                   )

      time.sleep(1)  # 1 second delay







