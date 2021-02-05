#!/usr/bin/python

import os,sys
import signal
from utils.utility import exception


SIGNAL ={
    1: "SIGHUP", # Action terminate process
    2: "SIGINT", # Action terminate process
    6: "SIGABRT",
    9: "SIGKILL", # Terminate process
    15: "SIGTERM" # Action termination process
}

class SignalHandler:

    def __init__(self):
        self.registered_functions = []

    def initiate(self):
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGABRT, self.handler)
        #signal.signal(signal.SIGBUS, self.handler)
        #signal.signal(signal.SIGKILL, self.handler)
        signal.signal(signal.SIGTERM, self.handler)

    @exception
    def handler(self,signal,frame):

        if signal in SIGNAL:
            print("INFO: Received {} Signal ... ".format(SIGNAL[signal]))
        else:
            print("INFO: Received {} Signal ... ".format(signal))
        for func in self.registered_functions:
            func()
        print("INFO: All functions are done, exiting ...")
        sys.exit()