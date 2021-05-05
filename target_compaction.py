#!/usr/bin/python
import os,sys
from utils.utility import exception, exec_cmd
import json
import time
import socket

"""
The target_compaction script start the compaction on each target node and wait for its completion.
"""

LOG_DIR="/var/log/dss"
TARGET_SRC_PATH="/usr/dss/nkv-target"
HOSTNAME = socket.gethostname()

class Compaction:
  def __init__(self, target_ip=""):
    self.logger = self.get_logger()
    self.nqn = self.get_subsystem_nqn()
    self.targets = target_ip
    self.status = {}
    self.finished_nqn_compaction = 0


  def __del__(self):
    if self.logger:
      self.logger.close()

  def get_logger(self):
    if not os.path.exists(LOG_DIR):
      os.makedirs(LOG_DIR)
    FH = None
    log_file_name = LOG_DIR + "/target_compaction.log"
    try:
      FH = open(log_file_name,"w")
    except Exception as e:
      print("EXCEPTION: {}".format(e))

    return FH


  def start(self):
    command = "sudo " + TARGET_SRC_PATH + "/scripts/dss_rpc.py -s /var/run/spdk.sock rdb_compact -n "
    for nqn in self.nqn:
      compaction_command = command + nqn
      #self.logger.write("INFO: Compaction started for - {}\n".format(nqn))
      ret, console  = exec_cmd(compaction_command, True, True)
      if ret == 0 and console:
        compaction_start_status = json.loads(console)
        if "result" in compaction_start_status and compaction_start_status["result"] == "STARTED":
          self.logger.write("INFO: Compaction started for NQN - {}\n".format(nqn))
          self.status[nqn] = False
        else:
          self.logger.write("ERROR: Failed to start compaction for NQN - {} \n {}\n".format(nqn, console))
      else:
        self.logger.write("ERROR: failed to start compaction for NQN - {}".format(nqn))
      
  

  def get_subsystem_nqn(self):
    #command = 'nvme list-subsys | grep  NQN | cut -d \'=\' -f 2'
    command = 'nvme list-subsys'
    ret, console = exec_cmd(command, True, True)
    nqn = []
    if ret == 0:
      lines = console.split()
      for line in lines:
        line =  line.decode('utf-8')
        if line.startswith('NQN'):
          subsystem_nqn = line.split("=")[-1]
          fields  = subsystem_nqn.split(":")
          if fields[-1].startswith(HOSTNAME):
            nqn.append(subsystem_nqn)
    print("INFO: Compaction should be initiated for the following NQNs \n {}".format(nqn))
    self.logger.write("INFO: Compaction should be initiated for the following NQNs \n {}\n".format(nqn))
    return nqn

  
  def get_status(self):
    command = "sudo " + TARGET_SRC_PATH + "/scripts/dss_rpc.py -s /var/run/spdk.sock rdb_compact --get_status -n "
    for nqn in self.nqn:
      status_command = command + nqn
      ret,console = exec_cmd(status_command, True, True)
      if ret == 0 and console:
        status = json.loads(console)
        if "result" in status and status["result"] == "IDLE":
          self.status[nqn] = True 
          self.finished_nqn_compaction +=1



if __name__ == "__main__":
  
  compaction = Compaction()
  compaction.start()
  while True:
    compaction.get_status()
    if compaction.finished_nqn_compaction >= len(compaction.status):
      compaction.logger.write("INFO: Compaction is finished!\n")
      break 
    time.sleep(1)

      
    
