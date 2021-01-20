#!/usr/bin/python

import os,sys
import time
import random
import argparse
import platform
from multiprocessing import Process,Queue,Value, Lock
import time
from task import Task, upload,get,delete

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
    self.logger_queue = kwargs.get("logger_queue", None)  # A multiprocessing queue

    self.operation_status_queue = kwargs.get("status_queue", None)  # Used by only client Application

    self.status = Value('i', 1)
    self.lock = Lock()
    self.process = None
    #print(kwargs)

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
    while self.process.is_alive() :
      time.sleep(1)
    #self.logger_lock.acquire()
    self.logger_queue.put("INFO: Worker-{} stopped!".format(self.id))
    #self.logger_lock.release()


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
          print("DEBUG: TaskQ Size-{}, worker-{}".format(self.task_queue.qsize(), self.id))
          self.logger_queue.put("DEBUG: TaskQ Size-{}, worker-{}".format(self.task_queue.qsize(),self.id))
          task = self.task_queue.get()
        except Exception as e:
          print("EXCEPTION:WORKER-{}:{}".format(self.id, e))
      else:
        #self.logger_queue.put("WWWWWWWWTTTTT ---->>DEBUG: TaskQ Size-{} , empty By worker-{}".format(self.task_queue.qsize(),self.id))
        pass

      if task:
        print("DEBUG: Task-{} is being processed by worker-{}".format(task.id, self.id))
        self.logger_queue.put("DEBUG: Task-{} is being processed by worker-{}".format(task.id, self.id))
        task.start(task_queue=self.task_queue,
                   index_data_queue=self.index_data_queue,
                   status_queue=self.operation_status_queue,
                   logger_queue=self.logger_queue)  # Start Task - May a PUT,LIST,DEL,GET operation

        # Worker update the
      time.sleep(2)  # 1 second delay
      #logger_queue.put("WWWWWWWW ---->>> INFO: WORKER: sample debug output from worker-{}".format(self.id))







