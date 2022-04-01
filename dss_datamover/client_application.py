#!/usr/bin/python

"""
# The Clear BSD License
#
# Copyright (c) 2022 Samsung Electronics Co., Ltd.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Samsung Electronics Co., Ltd. nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
from utils.config import Config, ClientApplicationArgumentParser
from utils.utility import exception, OPERATION_STATUS
from logger import MultiprocessingLogger
from multiprocessing import Process, Queue, Value, Lock, Manager
from worker import Worker
from task import Task
from nfs_cluster import NFSCluster
from datetime import datetime
import time
import socket
import sys
from socket_communication import ServerSocket, ClientSocket
import prctl

mgr = Manager()
index_buffer = mgr.dict()

BASE_DIR = os.path.dirname(__file__)
MONITOR_INACTIVE_WAIT_TIME = 1800  # 30 Mins
DEBUG_MESSAGE_INTERVAL = 600  # 10 Mins


class ClientApplication(object):

    def __init__(self, id, config):
        # ClientSharedResource.__init__(self)
        self.id = id
        self.config = config
        self.client_config = config.get("client", {})
        self.s3_config = config.get("s3_storage", {})
        self.host_name = socket.gethostname()
        self.operation = config["operation"]
        self.dryrun = config["dryrun"]
        self.debug = config.get("debug", False)
        self.ip_address_family = config.get("ip_address_family", "IPV4")
        self.fs_config = config.get("fs_config", {})

        # Message communication
        self.operation_data_queue = Queue()  # Hold file index data
        self.operation_data_lock = Lock()
        self.operation_status_queue = Queue()  # Hold file upload/list/del status
        self.operation_status_lock = Lock()
        self.index_data_receive_completed = Value('i', 0)
        self.index_data_receive_completed_lock = Lock()
        self.operation_status_send_completed = Value('i', 0)
        self.message_count = Value('i', 0)

        # self.operation_status_send_completed_lock = Lock()
        # Task
        self.task_queue = mgr.Queue()
        self.task_lock = Lock()
        self.task_id = 0

        # Workers
        self.workers = []
        self.workers_count = self.client_config.get("workers", 10)
        self.workers_alive = False

        # User credentials
        self.user_id = self.client_config.get("user_id", "ansible")
        self.password = self.client_config.get("password", "ansible")

        # Multiprocessing Logging
        self.logging_path = config["logging"].get("path", "/var/log/dss")
        self.logging_level = config["logging"].get("level", "INFO")
        if self.debug:
            self.logging_level = "DEBUG"
        self.logger_queue = Queue()
        self.logger_lock = Lock()
        self.logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
        self.logger = None

        # AWS log
        if 'aws' in config:
            self.aws_log_debug_val = int(config['aws'].get('awslib_log_debug', 0))
        else:
            self.aws_log_debug_val = 0

            # Message handling
        self.ip_address = self.config["ip_address"]
        self.port_index = config["port_index"]
        self.port_status = config["port_status"]
        self.socket = None
        self.stop_messaging = Value('i', 1)
        self.message_lock = Lock()

        # Mount NFS Share locally
        self.nfs_cluster = None
        self.nfs_share_list = Queue()

    def __del__(self):
        # TODO Stop all running process for abrupt shutdown
        # Make sure all NFS shares are unounted and removed
        while self.nfs_share_list.qsize() > 0:
            nfs_share = self.nfs_share_list.get()
            self.logger.info(
                "Un-mounting nfs-share {}:{}".format(nfs_share["nfs_cluster_ip"], nfs_share["nfs_share"]))
            self.nfs_cluster.umount(nfs_share["nfs_share"])

        self.stop_message()
        self.stop_workers()
        # self.nfs_cluster.umount_all()  # Un-mount all mounted shares
        self.stop_logging()

    def start(self):
        """
        Start client with the following functionalities
        :return:
        """

        # Start workers based on number of CPU available
        # Start messaging service
        self.start_logging()
        self.logger.info("Started Client Application for id:{} on node {}".format(self.id, self.host_name))
        self.nfs_cluster = NFSCluster(self.fs_config, "root", self.password, self.logger)
        if not self.start_workers():
            self.stop_logging()
            sys.exit("Workers were not started.")
        self.start_message()

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
        print("Make sure all workers has stopped the move to stopping logger")

        # TODO - Update self.nfs_cluster.local_mounts instead and call umount_all
        while self.nfs_share_list.qsize() > 0:
            nfs_share = self.nfs_share_list.get()
            self.logger.info(
                "Un-mounting nfs-share {}:{}".format(nfs_share["nfs_cluster_ip"], nfs_share["nfs_share"]))
            self.nfs_cluster.umount(nfs_share["nfs_share"])
        # self.nfs_cluster.umount_all()  # Un-mount all mounted shares
        self.stop_logging()

    def start_workers(self):
        """
        Launch workers on the node. The workers are independent process.
        Worker used multiprocessing logger to dump messages in a common location.
        :return:
        """
        index = 0
        while index < self.workers_count:
            w = Worker(id=index,
                       task_queue=self.task_queue,
                       status_queue=self.operation_status_queue,
                       index_data_queue=self.operation_data_queue,
                       logger=self.logger,
                       s3_config=self.s3_config,
                       skip_upload=self.config.get("skip_upload", False),
                       aws_log_debug_val=self.aws_log_debug_val)
            w.start()
            self.workers.append(w)
            index += 1

        # Check atleast one worker is RUNNING
        workers_started = False
        while True:
            s3_connection_failed = True
            for w in self.workers:
                if w.status.value == 1:
                    workers_started = True
                    break
                elif w.status.value != -1:
                    s3_connection_failed = False
            if s3_connection_failed or workers_started:
                break 
        if not workers_started:
            self.logger.fatal("Workers were not started exit ClientApplication-{}!".format(self.id))
        return workers_started

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
        self.process_index = Process(target=self.message_server_index)
        self.process_index.start()

        self.process_status = Process(target=self.message_server_status)
        self.process_status.start()

        self.logger.info("Started message server ...")

    @exception
    def stop_message(self):
        """
        Stop ZMQ server
        :return:
        """
        # First stop the loop in the process.
        self.stop_messaging.value = 0

        # recv function is a blocking call, hence terminate the process
        while self.process_index.is_alive():
            time.sleep(1)
            try:
                self.process_index.terminate()
                self.logger.info("Terminated MessageHandler - Index ...")
            except Exception as e:
                self.logger.execp("Terminated MessageHandler - Index - {} ".format(e))

        while self.process_status.is_alive():
            time.sleep(1)
            try:
                self.process_status.terminate()
                self.logger.info("Terminated MessageHandler - Status ...")
            except Exception as e:
                self.logger.excep("Terminated MessageHandler - Status  - {} ".format(e))

        self.logger.info("Stopped all MessageHandlers ... ")

    def message_server_index(self):
        """
        Message Handler process incoming file index
        index message:{"dir":<>,"files":["f1","f2"],"size":<size of all files>, "nfs_cluster":<IP>, "nfs_share":<path>}
        end message: {"indexing" : 1 }

        Once indexing is completed at Master application side, master send a end message to all clients to finish their
        operation.

        Termination of Process:
        - Gracefully: Once receive end message exit the loop, thus finish "file index" handling msg handler.
        - Gracefully: The client application stop the message handler with "stop_messaging" shared variable.
        - Forcefully, process gets terminated on receiving termination signal.

        ## TODO
        - Gracefully: Once receive end message exit the loop, thus finish "file index" handling msg handler.
        - Forcefully: process gets terminated on receiving termination signal. Need to add signal handler.
        :return:
        """
        name = "DM_client_message_server_index"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        try:
            socket =  ServerSocket(self.logger, self.ip_address_family)
            socket_index_address = "{}:{}".format(self.ip_address, self.port_index)
            socket.bind(self.ip_address, self.port_index)
            self.logger.info("Client Index-Monitor listening to - {}".format(socket_index_address))
        except Exception as e:
            self.logger.excep("Monitor-Index-Receiver, binding error - {} ".format(e))
            self.logger.fatal("** Clean all ClientApplication-{} running on the node **".format(self.id))
            return
        socket.accept()
        objects_count = 0
        start_time_no_response = 0 # Start time of no response from master
        debug_message_timer = datetime.now()
        while True:

            # Check messaging flag  and break the  , generally multiprocessing Queue and value is thread/process safe.
            if self.stop_messaging.value == 0:
                break
            try:
                message = socket.recv_json()
                if message:
                    # Check the end message arrived, exit loop
                    if "indexing_done" in message and message["indexing_done"]:
                        self.logger.info("Receiving Index-data completed. Closing Monitor-Index-Receiver! ")
                        self.index_data_receive_completed.value = 1
                        break

                    with self.message_count.get_lock():
                        self.message_count.value += 1
                    if self.debug:
                        self.logger.debug(
                            "Received Indexed MSG, Operation:{} , Prefix:{}".format(self.operation, message["dir"]))

                    is_index_data_added = False

                    if self.operation.upper() == "PUT" or self.operation.upper() == "TEST":
                        # Message validation, NFS Mounting if not already mounted on client node
                        if "nfs_cluster" in message and "dir" in message:
                            if self.config.get("master_node", None) and self.config["master_node"] == 1:
                                self.add_task(message)
                                is_index_data_added = True
                            else:
                                if self.nfs_mount(message["nfs_cluster"], message["nfs_share"]):
                                    self.add_task(message)
                                    is_index_data_added = True
                                else:
                                    self.logger.error("Issue with mounting! NFS-Share {}".format(message["nfs_share"]))
                        else:
                            self.logger.error("Bad formed message -{}".format(message))
                    elif self.operation.upper() == "DEL" or self.operation.upper() == "GET":
                        self.add_task(message)  # Add message to task queue to be consumed by workers.
                        is_index_data_added = True
                    elif self.operation.upper() == "LIST":
                        self.add_task(message)
                        continue

                    objects_count_under_prefix = len(message["files"])
                    objects_count += objects_count_under_prefix
                    if is_index_data_added:
                        # Create a Shared dictionary to check progress of PUT/DEL/GET operation
                        if message["dir"] in index_buffer:
                            index_buffer[message["dir"]] += len(message["files"])
                        else:
                            index_buffer[message["dir"]] = len(message["files"])

                    start_time_no_response = 0
                else:
                    if start_time_no_response == 0:
                        start_time_no_response = datetime.now()
                    else:
                        # Check for 30 Min inactivity
                        if (datetime.now() - start_time_no_response).seconds > MONITOR_INACTIVE_WAIT_TIME:
                            start_time_no_response = datetime.now()
                            self.logger.error("No message received from master in last 30 mins.")
                            # self.index_data_receive_completed.value = 1
                            # break

                # Debug message
                if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                    self.logger.info(
                        "Messages received from master-{}, Objects Count: {}".format(self.message_count.value,
                                                                                     objects_count))
                    debug_message_timer = datetime.now()

            except Exception as e:
                self.logger.excep("Monitor-Index - {}".format(e))

        self.logger.info(
            "Total messages received from master-{}, Objects Count: {}".format(self.message_count.value, objects_count))
        # Close socket connection and destroy context
        try:
            socket.close()
            self.logger.info("Monitor-Index-Receiver terminated gracefully !")
        except Exception as e:
            self.logger.excep("Monitor-Index-Receiver - {}".format(e))

    def add_task(self,message):
        """
        Add a task to task_queue to be consumed by a worker.
        :param message:
        :return:
        """
        task_data = {"dir": message["dir"], "files": message.get("files",[]), "size": message.get("size", 0)}
        task = Task(operation=self.operation,
                    data=task_data,
                    s3config=self.s3_config,
                    dryrun=self.dryrun,
                    user_id=self.user_id,
                    password=self.password,
                    dest_path=self.config.get("dest_path", None), # Used for GET and DataIntegrity test
                    distributed=self.config.get("distributed", False))  # Used for Distributed LISTing

        self.task_queue.put(task)  # Enqueue task to TaskQ

    def message_server_status(self):
        """
        The Monitor StatusHandler receive all the status from workers and send back to master for aggregation.
        Shut Down monitor:
        - Shutdown gracefully:
            - All the indexes are received by the IndexReceive monitor and that got terminated.
            - All the status have been received and PUSHed to master application.
            - Shutdown
        - Forcefully:
            - Set "stop_messaging" to exit loop and thus shutdown process.
            - If things get HUNG, it wait for 20 mins and if doesn't receive any status message then exit.
        :return:
        """
        name = "DM_client_message_server_status"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        try:
            socket_address = "tcp://{}:{}".format(self.ip_address, self.port_status)
            socket = ServerSocket(self.logger, self.ip_address_family)
            socket.bind(self.ip_address, self.port_status)
            socket.accept()
            self.logger.info("Monitor-Status Socket Address-{}".format(socket_address))
        except Exception as e:
            self.logger.excep("Monitor-Status socket binding error - {}".format(e))
            self.logger.fatal("** Clean all ClientApplication-{} running on the node **".format(self.id))
            return

        processed_objects_success_count = 0
        processed_objects_failure_count = 0
        received_status_message_count = 0  # Received from workers
        sent_status_message_count = 0 # status messages sent to master
        start_time_not_receiving_status_message = 0  # Time, monitor not receiving any status message from workers.
        debug_message_timer = datetime.now()

        while True:
            # Check messaging flag  and break the loop
            stop_messaging = self.stop_messaging.value
            if stop_messaging == 0:
                break

            status_message = {}
            try:
                if self.operation_status_queue.qsize() > 0:
                    status_message = self.operation_status_queue.get()  ## {"success": <>, "failure":<>}

                # Send response after adding data to operation data_queue as success
                if status_message:
                    self.logger.debug("PUSH - Sending message - {}".format(status_message))
                    received_status_message_count += 1
                    if socket.send_json(status_message):
                        sent_status_message_count += 1
                    if self.operation.upper() != "LIST":
                        processed_objects_success_count += status_message["success"]
                        processed_objects_failure_count += status_message["failure"]
                else:
                    if start_time_not_receiving_status_message == 0:
                        start_time_not_receiving_status_message = datetime.now()
                    else:
                        # Check for 30 Min inactivity, shutdown forcefully
                        if (datetime.now() - start_time_not_receiving_status_message).seconds > MONITOR_INACTIVE_WAIT_TIME:
                            start_time_not_receiving_status_message = datetime.now()
                            self.logger.error("No message received from workers in last 30 mins.")
                            # self.operation_status_send_completed.value = 1
                            # break

                # Debugging message, print in every 5 mints
                if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                    self.logger.info("Status messages sent to master : {}, Operation (success, failure) = ({}, {})"
                                     .format(received_status_message_count, processed_objects_success_count,
                                             processed_objects_failure_count))
                    debug_message_timer = datetime.now()
            except Exception as e:
                self.logger.excep("Monitor-StatusHandler {}".format(e))

            if self.index_data_receive_completed.value and self.message_count.value == received_status_message_count:
                self.logger.info("All operation status sent to Master. Closing Monitor-StatusHandler !")
                self.operation_status_send_completed.value = 1
                break
        # Processing status
        self.logger.info("Total status message received from workers: {}".format(received_status_message_count))
        self.logger.info("Total status message sent to master : {}".format(sent_status_message_count))
        self.logger.info("Total operation success count = {}".format(processed_objects_success_count))
        self.logger.info("Total operation failure count = {}".format(processed_objects_failure_count))

        # Close socket and destroy context.
        try:
            # Send end message to status_poller socket at master if all messages were not processed.
            # The exit message will help to exit the status_poller.
            if self.message_count.value > received_status_message_count:
                end_message = {"exit": True, "client": self.id}
                socket.send_json(end_message)
            time.sleep(1)
            socket.close()
            self.logger.info("Monitor-StatusHandler is terminated gracefully !")
        except socket.error as e:
            self.logger.excep("MOnitor-StatusHandler - {}".format(e))
        except Exception as e:
            self.logger.excep("Monitor-StatusHandler - {}".format(e))

    @exception
    def nfs_mount(self, nfs_cluster_ip=None, nfs_share=None):
        """
        Mount remote NFS share based on cluster dns/ip and NFS share
        :param nfs_cluster_ip: nfs cluster dns/ip
        :param path: nfs share , e.g - /dir
        :return:
        """
        if nfs_cluster_ip and nfs_share:
            # Don't mount if the nfs share all ready mounted.
            if nfs_cluster_ip in self.nfs_cluster.local_mounts and nfs_share in self.nfs_cluster.local_mounts[
                nfs_cluster_ip]:
                return True

            ret, console = self.nfs_cluster.mount(nfs_cluster_ip, nfs_share)
            if ret == 0:
                self.nfs_share_list.put({"nfs_cluster_ip": nfs_cluster_ip, "nfs_share": nfs_share})
                return True
            else:
                if ret is None:
                    return True
                self.logger.error(
                    "{}: NFS Mounting failed for {}:{}\n  {}".format(__file__, nfs_cluster_ip, path, console))

        return False

    def start_logging(self):
        """
        Start Multiprocessing logger
        :return:
        """
        self.logger = MultiprocessingLogger(self.logger_queue, self.logger_lock, self.logger_status)
        self.logger.config(self.logging_path, __file__, self.logging_level)
        self.logger.start()

    def stop_logging(self):
        """
        Stop multiprocessing logger
        :return:
        """
        self.logger.stop()

    def monitor_workers(self):
        """
        Monitors workers and take appropriate actions
        - When a worker in hung condition, it stop that worker and re-start
        - Log that message to client-log and send a message to Master.
        :return:
        """
        time.sleep(30)
        is_workers_alive = False
        for worker in self.workers:
            if self.operation_status_send_completed.value > 0:
                # self.logger.info("All the status messages have been sent out already.")
                return False
            # Check status of the worker.
            if worker.status.value and worker.is_hung(self.operation):
                # self.logger.warn("Worker-{} Shutting down!".format(worker.id))
                # worker.stop()
                # worker.start()
                pass
            if worker.status.value:
                is_workers_alive = True

        # If no worker alive, shutdown client monitors.
        if not is_workers_alive:
            self.logger.error("Workers are not alive!. Going to shutdown ClientApplication-{}".format(self.id))
            self.stop_message()
            return False
        return True



if __name__ == "__main__":
    params = ClientApplicationArgumentParser()

    if "config" not in params:
        params["config"] = BASE_DIR + "/config/config.json"
    config_obj = Config(params)
    config = config_obj.get_config()
    client_config = config.get("client", {})

    ca = ClientApplication(params["client_id"], config)
    ca.start()
    ca.logger.info("CONFIG:{}".format(config))
    ca.logger.info("Starting client application ...")

    while True:

        if ca.index_data_receive_completed.value and ca.operation_status_send_completed.value:
            # Un-mount local NFS shares
            while ca.nfs_share_list.qsize() > 0:
                nfs_share = ca.nfs_share_list.get()
                ca.logger.debug(
                    "Un-mounting nfs-share {}:{}".format(nfs_share["nfs_cluster_ip"], nfs_share["nfs_share"]))
                local_nfs_mount = os.path.abspath("/" + nfs_share["nfs_cluster_ip"] + "/" + nfs_share["nfs_share"])
                ret, console = ca.nfs_cluster.umount(local_nfs_mount)
                if ret == 0:
                    ca.logger.info(
                        "Un-mounted NFS share {} => {} successfully".format(nfs_share["nfs_share"], local_nfs_mount))
            # Stop message-handler
            ca.stop_message()  # May not be required, Already those process stopped.
            # Stop workers
            ca.stop_workers()
            ca.logger.info("All workers terminated !")
            break
        else:
            # if not ca.monitor_workers():
            #    break
            pass

        time.sleep(1)

    ca.logger.info("Stopping client application")
    ca.stop_logging()
    # ca.stop()
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
