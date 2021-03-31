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
import os,sys
from utils.config import Config, ClientApplicationArgumentParser
from utils.utility import exception
from logger import MultiprocessingLogger
from multiprocessing import Process,Queue,Value, Lock, Manager
from worker import Worker
from task import Task
from nfs_cluster import NFSCluster
import time
import zmq
import socket


mgr = Manager()
index_buffer = mgr.dict()


BASE_DIR=os.path.dirname(__file__)

class ClientApplication:

    def __init__(self, id, config):
        #ClientSharedResource.__init__(self)
        self.id = id
        self.config = config
        self.client_config = config.get("client",{})
        self.s3_config = config.get("s3_storage",{})
        self.host_name = socket.gethostname()
        self.operation = config["operation"]
        self.dryrun = config["dryrun"]

        # Message communication
        self.operation_data_queue = Queue()  # Hold file index data
        self.operation_data_lock = Lock()
        self.operation_status_queue = Queue() # Hold file upload/list/del status
        self.operation_status_lock = Lock()
        self.index_data_receive_completed = Value('i', 0)
        self.index_data_receive_completed_lock = Lock()
        self.operation_status_send_completed = Value('i', 0)

        #self.operation_status_send_completed_lock = Lock()
        # Task
        self.task_queue = mgr.Queue()
        self.task_lock = Lock()
        self.task_id = 0
        self.create_task_completed = Value('i', 0)

        # Workers
        self.workers = []
        self.workers_count = self.client_config.get("workers", 10)

        # Multiprocessing Logging
        self.logging_path = config.get("logging_path", "/var/log")
        self.logger_queue = Queue()
        self.logger_lock = Lock()
        self.logger = None

        # Message handling
        #self.ip_address = socket.gethostbyname(self.host_name)
        self.ip_address = self.config["ip_address"]
        self.port_index = config["port_index"]
        self.port_status = config["port_status"]
        self.socket = None
        self.stop_messaging = Value('i', 1)
        self.message_lock = Lock()

        # Mount NFS Share locally
        self.nfs_cluster = NFSCluster({}, self.logger_queue)
        self.nfs_share_list = Queue()


    def __del__(self):
        # TODO Stop all running process for abrupt shutdown

        # Make sure all NFS shares are unounted and removed
        while not self.nfs_share_list.empty():
            nfs_share = self.nfs_share_list.get()
            self.logger_queue.put("INFO: Un-mounting nfs-share {}:{}".format(nfs_share["nfs_cluster_ip"],nfs_share["nfs_share"]))
            self.nfs_cluster.umount(nfs_share["nfs_share"])
        """
        self.stop_workers()
        self.nfs_cluster.umount_all()  # Un-mount all mounted shares
        self.stop_logging()
        """




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
        This function can be used to forcefully shutdown all the running process.
        - Stop message handler
        - Stop CreateTask process
        - Stop Workers
        - NFS unmount
        :return: None
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
                       logger_queue=self.logger_queue,
                       s3_config=self.s3_config)
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
        #self.message_lock.acquire()
        self.stop_messaging.value = 0
        #self.message_lock.release()


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
        index message: {"dir":<>, "files":["f1","f2"], "size": <size of the set of files>}
        end message: {"indexing" : 1 }

        Once indexing is completed at Master application side, master send a end message to all clients to finish their
        operation.

        Termination of Process:
        - Gracefully: Once receive end message exit the loop, thus finish "file index" handling msg handler. #TODO
        - Gracefully: The client application stop the message handler with "stop_messaging" shared variable.
        - Forcefully, process gets terminated on receiving termination signal.

        ## TODO
        - Gracefully: Once receive end message exit the loop, thus finish "file index" handling msg handler.
        - Forcefully, process gets terminated on receiving termination signal. Need to add signal handler.
        :return:
        """
        context = zmq.Context()
        socket_index_address = "tcp://{}:{}".format(self.ip_address, self.port_index)
        socket = context.socket(zmq.REP)
        socket.bind(socket_index_address)
        self.logger_queue.put("INFO: Client Index-Monitor listening to - {}".format(socket_index_address))

        while True:

            # Check messaging flag  and break the  , generally multiprocessing Queue and value is thread/process safe.
            stop_messaging = self.stop_messaging.value
            #self.message_lock.release()
            if stop_messaging == 0:
                break

            # Index Msg: {"dir":<>, "files":["f1","f2"]}
            #self.logger_queue.put("DEBUG: Client-{}, Waiting to receive INDEX the message".format(self.id))
            message = {}
            ####
            try:
                received_response = socket.poll(timeout=1000)  # Wait 1 secs
                if received_response:
                    message = socket.recv_json()
                    #self.logger_queue.put("DEBUG: Received index message - {}".format(message))
                    # Check the end message arrived, exit loop
                    if "indexing_done" in message and message["indexing_done"]:
                        self.logger_queue.put("INFO: Receiving Index-data completed. Closing Monitor-Index-Receiver! ")
                        socket.send_json({"success": 1})
                        self.index_data_receive_completed.value = 1
                        break
                    #self.logger_queue.put("DEBUG: Received Indexed Message for Operation:{} , MSG:{}".format(self.operation, message["files"]))
                    is_index_data_added = False #
                    # Add indexing data to the "operation_data_queue".
                    if self.operation.upper() == "PUT":
                        ## Message validation, NFS Mounting if not already mounted on client node
                        if "nfs_cluster" in message and "dir" in message:
                            if self.config.get("master_node", None) and self.config["master_node"] == 1:
                                self.operation_data_queue.put(message)  ## Add index to operation_data_queue
                                socket.send_json({"success": 1})  # Send response back to MasterApp
                                is_index_data_added = True
                            else:
                                if self.nfs_mount(message["nfs_cluster"], message["dir"]):
                                    self.operation_data_queue.put(message) ## Add index to operation_data_queue
                                    socket.send_json({"success": 1})  # Send response back to MasterApp
                                    is_index_data_added = True
                                else:
                                    self.logger_queue.put("ERROR: Issue with mounting! ")
                                    ## Send the success/failure status to master
                                    socket.send_json({"success": 0, "ERROR": "Client-{} , Failed NFS -{} mounting".format(self.id,  message["dir"])})
                        else:
                            self.logger_queue.put("ERROR: Bad formed message -{}".format(message))
                            socket.send_json({"success": 0, "ERROR": "Client-{} , Bad Index MSG format -{}".format(self.id,  message)})

                    elif self.operation.upper() == "DEL" or self.operation.upper() == "GET":
                        self.operation_data_queue.put(message)  ## Add index to operation_data_queue
                        socket.send_json({"success": 1})  # Send response back to MasterApp
                        is_index_data_added = True

                    if is_index_data_added:
                        ## Create a Shared dictionary to check progress of PUT/DEL/GET operation
                        if message["dir"] in index_buffer:
                            index_buffer[message["dir"]] += len(message["files"])
                        else:
                            index_buffer[message["dir"]] = len(message["files"])
            except Exception as e:
                self.logger_queue.put("EXCEPTION: Monitor-Index - {}".format(e))

            #time.sleep(1)

        socket.close()
        self.logger_queue.put("INFO: Monitor-Index-Receiver terminated gracefully !")

    def message_server_status(self):
        context = zmq.Context()
        socket_address = "tcp://{}:{}".format(self.ip_address, self.port_status)
        socket = context.socket(zmq.PUSH)
        self.logger_queue.put("DEBUG: MessageHandler-Status Socket Address-{}".format(socket_address))
        socket.bind(socket_address)

        while True:

            # Check messaging flag  and break the loop
            stop_messaging = self.stop_messaging.value

            if stop_messaging == 0:
                break

            status_message = {}
            try:
                if not self.operation_status_queue.empty():
                    status_message = self.operation_status_queue.get()  ## {"success": <>, "failure":<>}
                # Send response after adding data to operation data_queue as success
                if status_message:
                    self.logger_queue.put("DEBUG: PUSH - Sending message - {}".format(status_message))
                    socket.send_json(status_message)
                    # Decrement index_buffer as those files have been processed.
                    if status_message["dir"] in index_buffer:
                        index_buffer[status_message["dir"]] -= ( status_message["success"] + status_message["failure"])

                # TODO - Optimization required.
                if self.index_data_receive_completed.value:
                    #self.logger_queue.put("YYYYYYYY -INDEX BUFFER:{}".format(index_buffer))
                    for dir_prefix, processed_file_count in index_buffer.items():
                        if processed_file_count == 0:
                            del index_buffer[dir_prefix]
                    """
                    if status_message and index_buffer[status_message["dir"]] == 0:
                        self.logger_queue.put("DEBUG: Remove dir IIIIIIII:{}:{}".format(status_message , index_buffer ))
                        try:
                            del index_buffer[status_message["dir"]]
                        except Exception as e:
                            self.logger_queue.put("EXCEPTION: Unable to remove shared index count : {}".format(e))
                    """

            except Exception as e:
                self.logger_queue.put("EXCEPTION: MessageHandler-Status {}".format(e))

            if self.index_data_receive_completed.value and   not index_buffer  :
                self.logger_queue.put("INFO: All operation status sent to Master. Closing Monitor-StatusHandler !")
                self.operation_status_send_completed.value = 1
                break

            #time.sleep(1)

        socket.close()
        self.logger_queue.put("INFO: Monitor-StatusHandler is terminated gracefully !")
        #self.logger_queue.put("XXXXXXXXXXXXXXX- index_buffer {}".format(index_buffer))

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
        - Terminate Loop:
          - Received all index data : self.index_data_receive_completed.value == 1
          - Created tasks for all received data: operation_data_queue.empty() == True

        :return:
        """
        self.logger_queue.put("INFO: CreateTask Process started !!!")

        while True:

            stop_messaging = self.stop_messaging.value
            if stop_messaging == 0:
                break

            index_data = {}
            # De-queue index data from "operation_data_queue" and create Task
            if not self.operation_data_queue.empty():
                index_data = self.operation_data_queue.get()


            # Validate Minio configuration to proceed
            if not self.s3_config:
                self.logger_queue.put("ERROR: S3 configuration information required! Stopping CreateTask process")
                break

            #print("Create Task - Index Data - {}".format(index_data))
            # Received index data {"dir": "/dir1/dir11", "files":["f1","f2"]}
            if index_data and "dir" in index_data and "files" in index_data:
                #index_data = index_data["index"]  # remove this
                #self.logger_queue.put("====>TASK started index data - {}:{}".format(index_data,self.operation))
                #print("CreateTask:{}".format(index_data))

                index =0
                while index < len(index_data["files"]):
                    try:
                        #task = Task(self.task_id, put, index_data[index:index+2], self.logger_queue)
                        task_data = {"dir": index_data["dir"], "files": index_data["files"][index:index + self.client_config.get("max_index_size", 2)]}
                        task = Task(operation=self.operation,
                                    data=task_data,
                                    s3config=self.s3_config,
                                    dryrun=self.dryrun,
                                    dest_path=self.config.get("dest_path",""))
                        self.task_queue.put(task)  # Enqueue task to TaskQ
                    except Exception as e:
                        print("Exception: create_task - {}".format(e))
                        self.logger_queue.put("Exception: Create_Task  - {}".format(e))
                    self.task_id +=1

                    index += self.client_config.get("max_index_size", 2)
            #time.sleep(1)

            # How to stop it?
            ## index_data receive is completed and  operation_data_queue is empty.
            if self.index_data_receive_completed.value and self.operation_data_queue.empty():
                self.logger_queue.put("INFO: Tasks have been created for all received index_data from master.")
                self.create_task_completed.value = 1
                break

        self.logger_queue.put("INFO: Create Task Monitor is terminated gracefully!")




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

    params["config"] = BASE_DIR + "/config/config.json"
    config_obj = Config(params)
    config = config_obj.get_config()
    client_config = config.get("client", {})
    logging_path = config.get("logging_path", "/var/log/dss")


    ca = ClientApplication(params.get("id", 1), config)
    ca.logger_queue.put("CONFIG:{}".format(config))
    ca.logger_queue.put("INFO: Starting client application ...")
    #print("INFO: Created client application ... ")
    ca.start()

    while True:
        if ca.index_data_receive_completed.value and ca.operation_status_send_completed.value:
            # Un-mount local NFS shares
            while not ca.nfs_share_list.empty():
                nfs_share = ca.nfs_share_list.get()
                ca.logger_queue.put(
                    "INFO: Un-mounting nfs-share {}:{}".format(nfs_share["nfs_cluster_ip"], nfs_share["nfs_share"]))
                ret, console = ca.nfs_cluster.umount(nfs_share["nfs_share"])
                if ret:
                    ca.logger_queue.put("ERROR: Unmount failed! {}".format(console))

            # Stop message-handler
            ca.stop_message() # May not be required, Already those process stopped.
            ca.logger_queue.put("INFO: Terminated all message handler !")

            # Stop create process, if not stopped yet
            if ca.create_task_completed.value == 0 and ca.process_task.is_alive():
                time.sleep(1)
                try:
                    ca.process_task.terminate()
                except Exception as e:
                    ca.logger_queue.put("EXCEPTION: Create-Task - {}".format(e))

                ca.logger_queue.put("INFO: Monitor Create-Task is terminated!")

            # Stop workers
            ca.stop_workers()
            ca.logger_queue.put("INFO: All workers terminated !")
            break

        time.sleep(1)

    ca.logger_queue.put("INFO: Stopping client application")
    ca.stop_logging()
    #ca.stop()
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