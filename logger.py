#!/usr/bin/python

import os,sys
import time
from utils.utility import  exception
from multiprocessing import Process, Queue, Value, Lock

"""
#TODO
- Use Logger as base class to start logging
- Allow application to start and stop as logger.start(), logger.stop()
"""



class Logger:

    def __init__(self):
        pass
    def start(self):
        pass
    def stop(self):
        pass


####
class MultiprocessingLogger:

    def __init__(self, path, file_name, queue, lock):
        self.path = path
        self.logfile = self.get_log_file(file_name)
        self.queue = queue
        self.logger_lock = lock
        self.stop_logging = Value('i', 0)
        self.stop_lock = Lock()

        self.process = None

    @exception
    def get_log_file(self, file_name):
        #print("Log File Name: {}".format(file_name))
        file_name = file_name.split('/')[-1]
        if not self.path:
            self.path = "/var/log"
        if not os.path.isdir(self.path):
            os.path.mkdir(self.path)
        file_name = file_name.replace("py", "log")
        return os.path.abspath(self.path + "/" + file_name)

    def start(self):
        if self.queue and self.logger_lock:
            self.process = Process(target=self.logging, args=(self.queue, self.logger_lock, self.stop_logging))
            self.process.start()

    def stop(self):

        ## DEBUG - Remove letter
        self.logger_lock.acquire()
        self.queue.put("INFO: LOGGER Stopping logging!")
        self.logger_lock.release()

        self.stop_lock.acquire()
        self.stop_logging.value = 1
        self.stop_lock.release()

        while self.process.is_alive():
            time.sleep(1)
            if self.queue.empty():
                try:
                    self.process.terminate()
                except Exception as e:
                    print("EXCEPTION: LOGGER - {}".format(e))

        print("INFO: LOGGER Stopped !!!")

    def status(self):
        self.stop_lock.acquire()
        s = self.stop_logging.value
        self.stop_lock.release()
        return s

    @exception
    def logging(self, queue, lock, stop_logging):
        try:
            print("Log file:{}".format(self.logfile))
            # Move existing log file to log1
            if os.path.exists(self.logfile):
                newfile = self.logfile + ".bak"
                if os.path.exists(newfile):
                    os.remove(newfile)
                os.rename(self.logfile, newfile)
            while True:
                fh = open(self.logfile, "a")

                while not queue.empty():
                    message = queue.get()
                    fh.write(message + "\n")  # Need to update to format the message based on severity
                self.stop_lock.acquire()
                stop = stop_logging.value
                self.stop_lock.release()
                if stop and queue.empty():
                    break
                time.sleep(1)
                fh.close()

        except Exception as e:
            print("Exception: {}".format(e))
        finally:
            fh.close()
        #print("Logger actually stopped!")










