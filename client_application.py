#!/usr/bin/python

import os,sys
from utils.config import Config, ClientApplicationArgumentParser
from utils.utility import exception
from logger import MultiprocessingLogger
from multiprocessing import Process,Queue,Value, Lock, Manager
from worker import Worker
from task import Task,upload
from nfs_cluster import NFSCluster
import time
import zmq
import socket


mgr = Manager()

"""
class ClientSharedResource:
    
    #All the shared resource used by all components at ClientApplication

    def __init__(self):
        #self.task_queue = Queue()
        self.operation_data_queue= Queue()
        self.operation_status_queue = Queue()
"""

class ClientApplication:

    def __init__(self, id, config, logging_path=None ):
        #ClientSharedResource.__init__(self)
        self.id = id
        self.config = config
        self.host_name = socket.gethostname()

        # Message communication
        self.operation_data_queue = Queue()  # Hold file index data
        self.operation_data_lock = Lock()
        self.operation_status_queue = Queue() # Hold file upload/list/del status
        self.operation_status_lock = Lock()
        # Task
        self.task_queue = mgr.Queue()
        self.task_lock = Lock()
        self.task_id = 0

        # Workers
        self.workers = []
        self.workers_count = config.get("workers", 10)

        # Multiprocessing Logging
        self.logging_path = logging_path
        self.logger_queue = Queue()
        self.logger_lock = Lock()
        self.logger = None

        # Message handling
        self.ip_address = socket.gethostbyname(self.host_name)
        self.port_index = "6000"
        self.port_status = "6001"
        self.socket = None
        self.stop_messaging = Value('i', 1)
        self.message_lock = Lock()

        # Mount NFS Share locally
        self.nfs_cluster = NFSCluster({}, self.logger_queue)
        self.nfs_share_list = Queue()


    def __del__(self):
        self.stop_workers()
        self.nfs_cluster.umount_all()  # Un-mount all mounted shares
        self.stop_logging()




    def start(self):
        """
        Start client with the following functionalities
        :return:
        """

        # Start workers based on number of CPU available
        # Start messaging service
        self.start_logging()
        self.logger_queue.put("INFO: Started Client Application for id:{} on node {}".format(self.id, self.host_name))
        self.start_workers()
        self.start_message()
        self.start_task_creation()


    def stop(self):
        """
        Stop the workers
        Stop message handling agent
        :return:
        """
        self.stop_workers()
        self.stop_message()
        while self.process_task.is_alive():
            time.sleep(1)
            try:
                self.process_task.terminate()
            except Exception as e:
                self.logger_queue.put("EXCEPTION: Task - {}".format(e))
        print("Make sure all workers has stopped the move to stopping logger")
        self.logger_queue.put("Start unmounting all NFS share")
        self.logger_queue.put("PPPPPPPPPPPPPPP: ClientApplication: NFS MountedShares-{}".format(self.nfs_cluster.local_mounts))
        ## TODO - Update self.nfs_cluster.local_mounts instead and call umount_all
        while not self.nfs_share_list.empty():
            nfs_share = self.nfs_share_list.get()
            self.logger_queue.put("INFO: Un-mounting nfs-share {}:{}".format(nfs_share["nfs_cluster_ip"],nfs_share["nfs_share"]))
            self.nfs_cluster.umount(nfs_share["nfs_share"])
        #self.nfs_cluster.umount_all()  # Un-mount all mounted shares
        self.stop_logging()

    def start_workers(self):
        """
        Launch workers on the node. The workers are independent process.
        Worker used multiprocessing logger to dump messages in a common location.
        :return:
        """
        index = 0
        #self.logger.write("DEBUG: Starting workers for client-{}\n".format(self.id))
        while index < self.workers_count:
            w = Worker(id=index,
                       task_queue=self.task_queue,
                       status_queue=self.operation_status_queue,
                       index_data_queue=self.operation_data_queue,
                       logger_queue=self.logger_queue)
            #self.logger.write("DEBUG: Starting worker-{}\n".format(index))
            w.start()
            self.workers.append(w)
            index += 1
        #self.logger.write("DEBUG: All workers started\n")


    def stop_workers(self, id=None):
        """
        Stop worker/s
        :param id: worker id
        :return:
        """
        index = 0
        while index < self.workers_count:
            w = self.workers[index]
            if w.get_status():
                w.stop()
            index += 1

    def start_message(self):
        """
        Start ZMQ Server
        :return:
        """
        print("INFO: Starting messaging system ...")

        #self.message_server_index()
        self.process_index = Process(target=self.message_server_index )
        self.process_index.start()

        self.process_status = Process(target=self.message_server_status)
        self.process_status.start()

        self.logger_queue.put("INFO: Started message server ...")

    @exception
    def stop_message(self):
        """
        Stop ZMQ server
        :return:
        """
        # First stop the loop in the process.
        self.message_lock.acquire()
        self.stop_messaging.value = 0
        self.message_lock.release()


        print("Waiting for message process to finish ....")

        ## recv function is a blocking call, hence terminate the process
        while self.process_index.is_alive():
            time.sleep(1)
            try:
                self.process_index.terminate()
                print("INFO: Terminated MessageHandler - Index ...")
            except Exception as e:
                print("EXCEPTION: Terminated MessageHandler - Index - {} ".format(e))

        while self.process_status.is_alive():
            time.sleep(1)
            try:
                self.process_status.terminate()
                print("INFO: Terminated MessageHandler - Status ...")
            except Exception as e:
                print("EXCEPTION: Terminated MessageHandler - Status  - {} ".format(e))

        print("DEBUG: Stopped all MessageHandler ...")
        self.logger_queue.put("INFO: Stopped all MessageHandlers ... ")

    def message_server_index(self):
        """
        Message Handler process incoming file index
        ## TODO
        - Once Process
        :return:
        """
        context = zmq.Context()
        socket_index_address = "tcp://{}:{}".format(self.ip_address, self.port_index)
        socket = context.socket(zmq.REP)
        #print("GGGG Binding to -{}".format(socket_index_address))
        socket.bind(socket_index_address)

        while True:

            # Check messaging flag  and break the loop
            #self.message_lock.acquire()
            stop_messaging = self.stop_messaging.value
            #self.message_lock.release()
            if stop_messaging == 0:
                break

            # Index Msg: {"dir":<>, "files":["f1","f2"], "size": <size of the set of files>}
            self.logger_queue.put("DEBUG: Client-{}, Waiting to receive INDEX the message - ".format(self.id))
            message = socket.recv_json()

            print("RES: Received Message -- {}".format(message))
            self.logger_queue.put("INFO: Received index message - {}".format(message))

            # Steps:
            ## TODO Validate message
            ## Mount the NFS share if not already mounted on client node.
            ## Add index to index queue/operation_data_queue
            try:
                if self.nfs_mount(message["nfs_cluster"], message["dir"]):
                    self.operation_data_queue.put(message)
                    socket.send_json({"success": 1}) # Send response after adding data to operation_data_queue as success
                else:
                    self.logger_queue.put("DEBUG: Issue with mounting! ")
                    socket.send_json({"success": 0})

            except Exception as e:
                self.logger_queue.put("EXCEPTION: MessageHandler-Index {}".format(e))
                socket.send_json({"success": 0})
            time.sleep(1)

        socket.close()
        print("Closing index socket ...")

    def message_server_status(self):
        context = zmq.Context()
        socket_address = "tcp://{}:{}".format(self.ip_address, self.port_status)
        socket = context.socket(zmq.PUSH)
        self.logger_queue.put("DEBUG: MessageHandler-Status Socket Address-{}".format(socket_address))
        socket.bind(socket_address)

        while True:

            # Check messaging flag  and break the loop
            self.message_lock.acquire()
            stop_messaging = self.stop_messaging.value
            self.message_lock.release()
            if stop_messaging == 0:
                break

            status_message = {}
            try:
                self.operation_status_lock.acquire()
                if not self.operation_status_queue.empty():
                    status_message = self.operation_status_queue.get()  ## {"success": <>, "failure":<>}
                self.operation_status_lock.release()
                # Send response after adding data to operation data_queue as success
                #print("PUSH: Sending message - {}".format(status_message))
                if status_message:
                    self.logger_queue.put("PUSH: Sending message - {}".format(status_message))
                    socket.send_json(status_message)
            except Exception as e:
                self.logger_queue.put("EXCEPTION: MessageHandler-Status {}".format(e))
                #socket.send_json(status_message)
            time.sleep(1)

        socket.close()
        self.logger_queue.put("INFO: MassageHandler - Closing status socket ...!")

    @exception
    def nfs_mount(self, nfs_cluster_ip=None, path=None):
        """
        Mount remote NFS share based on cluster dns/ip and NFS share
        :param nfs_cluster_ip: nfs cluster dns/ip
        :param path: nfs share , e.g - /dir
        :return:
        """
        if nfs_cluster_ip and path:
            nfs_share = "/" + (path.lstrip()).split("/")[1]
            # Don't mount if the nfs share all ready mounted.
            if nfs_cluster_ip in self.nfs_cluster.local_mounts and nfs_share in self.nfs_cluster.local_mounts[nfs_cluster_ip]:
                return True

            ret,console = self.nfs_cluster.mount(nfs_cluster_ip,nfs_share)
            if ret == 0:
                print("INFO: Mounted NFS share {}:{}".format(nfs_cluster_ip,nfs_share))
                self.logger_queue.put("INFO: Mounted NFS share {}:{}".format(nfs_cluster_ip,nfs_share))
                self.nfs_share_list.put({"nfs_cluster_ip": nfs_cluster_ip, "nfs_share": nfs_share})

                if nfs_cluster_ip not in self.nfs_cluster.local_mounts:
                    self.nfs_cluster.local_mounts[nfs_cluster_ip] = [nfs_share]
                else:
                    self.nfs_cluster.local_mounts[nfs_cluster_ip].append(nfs_share)
                self.nfs_cluster.mounted = True

                return True
            else:
                print("ERROR:NFS mounting failed \n {}".format(console))
                self.logger_queue.put("ERROR:{}: NFS Mounting failed for {}:{}\n  {}".format(__file__,nfs_cluster_ip,path,console))


        return False

    def start_logging(self):
        """
        Start Multiprocessing logger
        :return:
        """
        self.logger = MultiprocessingLogger(self.logging_path, __file__, self.logger_queue, self.logger_lock)
        self.logger.start()

    def stop_logging(self):
        """
        Stop multiprocessing logger
        :return:
        """
        if not self.logger.status():
            self.logger.stop()


    def start_task_creation(self):
        self.process_task = Process(target=self.create_task, args=(self.task_queue,))
        self.process_task.start()

    def create_task(self, task_queue):
        """
        Create a TASk and add that task to task_queue.
        - Loop
        - Read data from NFS Index Queue/ operation_data_queue
        - Create a task with at least 5 file index ( Variable )
        - If more than 5 task exist in the message, then split them to more tasks.

        :return:
        """
        self.logger_queue.put("====>TASK started !!!")
        print("Create Task Function ....")
        while True:

            #self.message_lock.acquire()
            stop_messaging = self.stop_messaging.value
            #self.message_lock.release()
            if stop_messaging == 0:
                break

            index_data = {}
            #self.operation_data_lock.acquire()
            if not self.operation_data_queue.empty():
                index_data = self.operation_data_queue.get()
            #self.operation_data_lock.release()
            #print("Create Task - Index Data - {}".format(index_data))
            # Received index data {"dir": "/dir1/dir11", "files":["f1","f2"]}
            if index_data and "dir" in index_data and "files" in index_data:
                #index_data = index_data["index"]  # remove this
                self.logger_queue.put("====>TASK started index data - {}".format(index_data))
                print("CreateTask:{}".format(index_data))

                index =0
                while index < len(index_data["files"]):
                    try:
                        #task = Task(self.task_id, put, index_data[index:index+2], self.logger_queue)
                        task_data = {"dir": index_data["dir"], "files": index_data["files"][index:index + 2]}
                        task = Task(operation="upload", data=task_data)

                        print("====>TASK has been created GGGGGGG- {}".format(index_data["files"][index:index+2]))
                        #self.logger_queue.put("====>TASK has been created - {}".format(index_data[index:index+2]))
                        self.task_queue.put(task)  # Enqueue task to TaskQ

                        #self.logger_queue.put("====>TaskQ Size - {}".format(self.task_queue.qsize()))
                        print("=====>>TaskQ Size-{}".format(self.task_queue.qsize()))
                        ## REMOVE  --- TEST CODE
                        #self.operation_status_queue.put({"success": 2, "failure": 0})
                    except Exception as e:
                        print("Exception: create_task - {}".format(e))
                        self.logger_queue.put("Exception: Create_Task  - {}".format(e))
                    self.task_id +=1

                    index += 2
            time.sleep(1)




"""
# TODO 
Replace above client messaging through this class.
"""
class ClientMessage:

    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass




if __name__ == "__main__":
    params = ClientApplicationArgumentParser()

    params["config"] = "/home/somnath.s/work/FB_Data_Mover/AmazonSDK/config/config.json"
    config_obj = Config(params)
    config = config_obj.get_config()
    client_config = config.get("client", {})
    logging_path = config.get("logging_path", "/var/log")

    print(config)

    ca = ClientApplication(params.get("id", 1), client_config, logging_path)


    ca.logger_queue.put("INFO: Created client application ...")
    print("INFO: Created client application ... ")
    ca.start()
    time.sleep(30)
    ca.logger_queue.put("INFO: Stopping client application")
    ca.stop()

    print("INFO: Stopped client application")


    # Stop execution
    ## Received the instruction from master that all data has been sent
    ## Process all outstanding data from message queue
    ## Stop all worker processes
    ## Stop Status process which sends message to Master once all the status is send.
    ### Send a message to master that closing client application.
    ## Stop logger process
    ### Check all outstanding logging message is written to a file./var/log/client_application.log

    ### 10.110.