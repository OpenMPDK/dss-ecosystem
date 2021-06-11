#!/usr/bin/
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
from utils.utility import exception, exec_cmd, remoteExecution, get_s3_prefix, progress_bar
from utils.config import Config, commandLineArgumentParser, CommandLineArgument
from utils.signal_handler import SignalHandler
from logger import  MultiprocessingLogger

from multiprocessing import Process,Queue,Value, Lock, Manager
from worker import Worker
from monitor import Monitor
from nfs_cluster import NFSCluster
from task import Task, iterate_dir
import time
import zmq
import signal
from datetime import datetime
import json


manager = Manager()

class Master(object):
    """
    Master application initiate
    -
    """

    def __init__(self, operation, config):
        self.config = config
        self.client_ip_list = config.get("tess_clients_ip",[])
        self.workers_count = config["master"]["workers"]
        self.max_index_size = config["master"]["max_index_size"]
        self.workers = []
        self.clients = []
        self.task_queue = Queue()
        self.task_lock = Lock()

        self.operation=operation
        self.client_user_id = config["client"]["user_id"]
        self.client_password = config["client"]["password"]

        self.s3_config = config.get("s3_storage", {})
        self.nfs_config =  self.config.get("nfs_config", {})

        ## Logging
        self.logging_path =  "/var/log/dss"
        self.logging_level = "INFO"
        if "logging" in config:
            self.logging_path = config["logging"].get("path", "/var/log/dss")
            self.logging_level = config["logging"].get("level", "INFO")
        self.logger       = None
        self.logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
        self.logger_queue = Queue()
        self.logger_lock  = Lock()

        self.lock = Lock()

        # Operation PUT/GET/DEL/LIST
        self.index_data_queue = Queue()  # Index data stored by workers
        self.index_data_lock = Lock()
        self.index_data_generation_complete = Value('i', 0)   ##TODO - Set value when all indexing is done.
        self.indexing_started_flag = Value('i', False)

        # Operation LIST
        self.prefix = config.get("prefix", None)
        self.listing_progress = manager.dict()
        self.listing_progress_lock = manager.Lock()
        self.listing_status = Value('i', 0) # [0,1,2] => ['NOT STARTED', 'STARTED', 'COMPLETED']
        self.listing_only = Value('b', False)
        self.listing_aggregation_status = Value('i', 0)
        self.listing_objectkey_queue = Queue()

        # Status Progress
        self.index_data_count = Value('i', 0)  # File Index count shared between worker process.
        self.operation_start_time = None
        self.operation_end_time = None

        # Keep track of progress of hierarchical indexing.
        self.progress_of_indexing = manager.dict()
        self.progress_of_indexing_lock = manager.Lock()

        # Hierarchical indexing of leaf directory, to be used for listing.
        self.indexing_key = manager.dict()
        self.indexing_key = manager.Lock()

        # NFS shares
        self.nfs_shares = []

        # Unit TestCase
        self.testcase_passed = Value('b', False)




    def __del__(self):
        # Stop workers
        #self.stop_workers()
        # Stop Monitor
        # Stop Messaging
        # Stop clients
        #self.stop_logging()
        pass
        # Remove local NFS share


    def start(self):
        """
        Start the master application with the following stuff
        - Start logger
        - Launch workers
        - Spawn clients
        - Start Monitor
        - Start Messaging ( File indexing message channel, progress status )
        :return:
        """
        self.start_logging()
        self.logger.info("Performing {} operation".format(self.operation))
        self.start_workers()
        self.operation_start_time = datetime.now()
        if not self.operation.upper() == "LIST":
            self.spawn_clients()

        if self.operation.upper() == "PUT" or self.operation.upper() == "TEST":
            self.start_indexing()
        if self.operation.upper() == "LIST" or self.operation.upper() == "DEL" or self.operation.upper() == "GET":
            self.start_listing()

        self.start_monitor()

        self.logger.info("DataMover running with \"{}\"  S3 client".format(self.s3_config["client_lib"]))

    def stop(self):
        """
        Stop master and its all clients and their application.
        - Send notification to clients to stop
        - Stop all the workers
        - Stop Monitor
        - Stop messaging systems.
        :return:
        """
        self.stop_workers()
        self.stop_clients()
        self.stop_monitor()
        self.nfs_cluster_obj.umount_all()
        self.stop_logging()

    def start_workers(self):
        """
        Launch workers
        :return:
        """
        index = 0
        while index < self.workers_count:
            w = Worker(id=index,
                       task_queue=self.task_queue,
                       logger=self.logger,
                       index_data_queue=self.index_data_queue,
                       progress_of_indexing= self.progress_of_indexing,
                       progress_of_indexing_lock=self.progress_of_indexing_lock,
                       index_data_count=self.index_data_count,
                       listing_progress=self.listing_progress,
                       listing_progress_lock=self.listing_progress_lock,
                       s3_config=self.s3_config,
                       indexing_started_flag=self.indexing_started_flag,
                       listing_status=self.listing_status,
                       listing_only=self.listing_only,
                       listing_objectkey_queue=self.listing_objectkey_queue
                       )
            w.start()
            self.workers.append(w)
            index +=1

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
        self.logger.info("Stopped all the workers running with MasterApplication! ")

    def spawn_clients(self):
        """
        Spawn the clients
        :return:
        """
        index =0
        for client_ip in self.client_ip_list:
            client = Client(index,
                            client_ip,
                            self.operation,
                            self.logger,
                            self.config)
            client.start()
            self.logger.info("Started ClientApplication-{} at node - {}".format(client.id,client_ip))
            self.clients.append(client)
            index +=1


    def stop_clients(self):
        """
        Stop all the client application associated with this master application
        :return:
        """
        for client in self.clients:
            client.stop()

    def start_monitor(self):
        """
        Monitor the progress of the operation( PUT,DEL,LIST)
        :return:
        """
        self.monitor = Monitor(clients=self.clients,
                               config=self.config,
                               index_data_queue=self.index_data_queue,
                               index_data_lock=self.index_data_lock,
                               index_data_generation_complete=self.index_data_generation_complete,
                               index_data_count=self.index_data_count,
                               logger=self.logger,
                               operation=self.operation,
                               operation_start_time=self.operation_start_time,
                               listing_status=self.listing_status,
                               listing_aggregation_status=self.listing_aggregation_status,
                               listing_objectkey_queue=self.listing_objectkey_queue,
                               testcase=self.testcase_passed
                               )
        self.monitor.start()
    def stop_monitor(self):
        """
        Stop all monitors forcefully.
        :return:
        """
        if self.monitor:
            self.monitor.stop()

    def start_logging(self):
        """
        Start Multiprocessing logger
        :return:
        """
        self.logger = MultiprocessingLogger(self.logger_queue,
                                            self.logger_lock,
                                            self.logger_status
                                            )
        self.logger.config(self.logging_path,
                           __file__,
                           self.logging_level)
        self.logger.start()
        self.logger.info("Started Logger with {} mode!".format(self.logging_level))
        #print("INFO: Logger started, (status -> {})".format(self.logger.status()))

    def stop_logging(self):
        """
        Stop multiprocessing logger
        :return:
        """
        self.logger.stop()


    def start_indexing(self):
        """
        Indexing is required for PUT operation
        :return:
        """
        # Fist Mount all NFS share locally
        self.nfs_cluster_obj = NFSCluster(self.config.get("nfs_config", {}), "root", "" , self.logger)
        self.nfs_cluster_obj.mount_all()

        # Create first level task for each NFS share
        local_mounts = self.nfs_cluster_obj.get_mounts()
        #print(local_mounts)
        for ip_address, nfs_shares in local_mounts.items():
            self.logger.info("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
            self.nfs_shares.extend(nfs_shares)
            for nfs_share in nfs_shares:
                #print("DEBUG: Creating task for {}".format(nfs_share))
                nfs_share_mount = os.path.abspath("/" + ip_address + "/" + nfs_share)
                task = Task(operation="indexing",
                            data=nfs_share_mount,
                            nfs_cluster=ip_address,
                            nfs_share=nfs_share,
                            max_index_size=self.max_index_size)
                self.task_queue.put(task)


    def start_listing(self):
        self.logger.info("Started Listing operation!")
        bad_prefix_no_listing = True
        # Create a Task based on prefix
        if self.prefix:
            self.logger.debug("Creating LIST task for prefix - {}".format(self.prefix))
            for prefix in get_s3_prefix(self.logger, self.config.get("nfs_config", {}), self.prefix):
                bad_prefix_no_listing = False
                task= Task(operation="list",
                       data={"prefix": prefix},
                       s3config=self.config["s3_storage"],
                       max_index_size=self.config["master"].get("max_index_size", 10)
                       )
                self.task_queue.put(task)
        else:
            for prefix in get_s3_prefix(self.logger, self.nfs_config):
                bad_prefix_no_listing = False
                task = Task(operation="list",
                            data={"prefix": prefix},
                            s3config=self.config["s3_storage"],
                            max_index_size=self.config["master"].get("max_index_size", 10)
                            )
                self.task_queue.put(task)

        if bad_prefix_no_listing:
            self.logger.error("LISTING failure!")
            self.listing_status.value = 1


    def compaction(self):
        # Spawn the process on target node and wait for response.
        command = "python3 /usr/dss/nkv-datamover/target_compaction.py"
        compaction_status = {}
        start_time = datetime.now()
        for client_ip in self.config["dss_targets"]:
            #print("INFO: Started Compaction for target-ip:{}".format(client_ip))
            self.logger.info("Started Compaction for target-ip:{}".format(client_ip))
            ssh_client_handler, stdin, stdout, stderr = remoteExecution(client_ip, self.client_user_id, self.client_password,command)
            compaction_status[client_ip] = {"status": False, "ssh_remote_client": ssh_client_handler, "stdout":stdout,"stderr":stderr}

        while True:
            is_compaction_done = True
            progress_bar("Compaction in Progress")
            for client_ip in compaction_status:
                if "status" in compaction_status[client_ip] and  compaction_status[client_ip]["status"]:
                    continue
                if "ssh_remote_client" in compaction_status[client_ip] and compaction_status[client_ip]["ssh_remote_client"]:
                    if "stdout" in compaction_status[client_ip] and compaction_status[client_ip]["stdout"]:
                        status = compaction_status[client_ip]["stdout"].channel.exit_status_ready()
                        if status:
                            #print("\nINFO: Compaction is finished for - {}".format(client_ip))
                            self.logger.info("Compaction is finished for - {}".format(client_ip))
                            compaction_status[client_ip]["status"] = True
                            compaction_status[client_ip]["ssh_remote_client"].close()
                        else:
                            is_compaction_done = False
            if is_compaction_done:
                break


        compaction_time = (datetime.now() - start_time).seconds
        self.logger.info("Total Compaction time - {} seconds".format(compaction_time))








class Client(object):

    def __init__(self, id, ip, operation, logger, config):
        """
        Client object initiation
        :param id: A id for each ClientApplication
        :param ip: IP address of the node in which ClienApplication will run
        :param operation: PUT|GET|DEL|LIST
        :param logger: A Logger with shared Queue to be used by multiple process/workers.
        :param config: configuration.
        """
        self.id = id
        self.ip = ip
        self.operation=operation
        self.config = config
        # Logger
        self.logger = logger
        self.debug = config.get("debug", False)

        # Variable from config
        self.username = "ansible"
        self.password = "ansible"
        if "client" in config:
            if "user_id" in config["client"]:
                self.username = config["client"]["user_id"]
            if "password" in config["client"]:
                self.password = config["client"]["password"]
        self.status = None
        self.ssh_client_handler = None
        self.master_ip_address = config["master"]["ip_address"]
        self.dryrun = config.get("dryrun",False)
        self.destination_path = config.get("dest_path","") # Only to be used for GET operation

        self.env_gcc_source = None
        self.env_gcc_required =  True
        if "environment" in config and "gcc" in config["environment"]:
            self.env_gcc_required = config["environment"]["gcc"].get("required", True)
            if self.env_gcc_required:
                self.env_gcc_source = config["environment"]["gcc"].get("source", "/usr/local/bin/setenv-for-gcc510.sh")
                self.logger.debug("Sourcing GCC environment from {} for GCC v-{}".format(self.env_gcc_source,
                                    config["environment"]["gcc"].get("version","GCC-VERSION-NOT-SPECIFIED")))
                #self.logger.info("Using GCC v-{} for dss_client library ".format(config["environment"]["gcc"].get("version","GCC-VERSION-NOT-SPECIFIED")))

        # Messaging service configuration
        if "message" in config:
            self.port_index = config["message"].get("port_index", "6000")  # Need to configure from configuration file.
            self.port_status =config["message"].get("port_status", "6001")
        else:
            self.port_index = "6000"
            self.port_status = "6001"
        self.socket_index = None
        self.socket_status = None

        # Remote output
        self.remote_execution_command =None
        self.remote_stdin =None
        self.remote_stdout = None
        self.remote_stderr = None

        # Execution status
        self.status = False

    def __del__(self):
        """
        Receive the PID of the remote process, so that forcefully remote process can be terminated.
        :return:
        """
        if not self.status:
            self.stop()

    def start(self):
        """
        Remote execution of client application ("client_application.py")
        :return:
        """
        # Setup
        #self.setup()
        #print("INFO: Starting ClientApplication-{} on node {}".format(self.id,self.ip))
        #self.logger.info("INFO: Starting ClientApplication-{} on node {}".format(self.id,self.ip))
        command = "python3 /usr/dss/nkv-datamover/client_application.py " + \
                                                                " --client_id {} ".format(self.id) + \
                                                                " --operation {} ".format(self.operation) + \
                                                                " --ip_address {} ".format(self.ip) + \
                                                                " --port_index {} ".format(self.port_index) + \
                                                                " --port_status {}  ".format(self.port_status)

        if self.operation.upper() == "GET" or self.operation.upper() == "TEST":
            command += " --dest_path {} ".format(self.destination_path)
        if self.master_ip_address == self.ip:
            command += " --master_node "
        if self.config.get("config", False):
            command += " --config {} ".format(self.config["config"])
        if self.dryrun:
            command += " --dryrun "
        if self.debug:
            command += " --debug "

        if self.env_gcc_required and self.env_gcc_source:
            command = "sh -c \" source {} && {} \"".format(self.env_gcc_source, command)
        self.ssh_client_handler, stdin,stdout,stderr = remoteExecution(self.ip, self.username , self.password, command)
        self.remote_stdin = stdin
        self.remote_stdout = stdout
        self.remote_stderr = stderr

    def remote_client_status(self):
        if self.ssh_client_handler:
            if self.remote_stdout:
                self.status = self.remote_stdout.channel.exit_status_ready()
                if self.status:
                    #print("INFO: Remote ClientApplication-{} terminated!".format(self.id))
                    self.logger.info("Remote ClientApplication-{} terminated!".format(self.id))
                return self.status
        return False


    def remote_client_exit_status(self):
        """
        It is blocking call.
        :return: exit status integer value
        """
        if self.ssh_client_handler:

            exit_status = self.remote_stdout.channel.recv_exit_status()
            stdout_lines = self.remote_stdout.readlines()
            stderr_lines = self.remote_stderr.readlines()
            self.logger.debug("Client-{} Remote execution status: {}".format(self.id,exit_status))
            if exit_status:
                #print("DEBUG: Client-{} \n STDOUT {}".format(self.id,stdout_lines))
                self.logger.debug("Client-{} \n STDOUT {}".format(self.id,stdout_lines))
            if stderr_lines:
                #print("ERROR: Client-{} \n STDERR {}".format(self.id,stderr_lines))
                self.logger.error("Client-{} \n STDERR {}".format(self.id,stderr_lines))
            self.ssh_client_handler.close()

        return exit_status


    def stop(self):
        """
        Send a message to the client node through ZMQ.
        # TODO
        Need to terminate forcefully.
        :return:
        """
        self.logger.info("Stopping client-{}".format(self.id))
        if self.ssh_client_handler:

            status = self.remote_stdout.channel.recv_exit_status()
            stdout_lines = self.remote_stdout.readlines()
            stderr_lines = self.remote_stderr.readlines()
            self.logger.info("Client-{} Remote execution status-{}".format(self.id,status))
            if status:
                self.logger.debug("Client-{} \n STDOUT {}".format(self.id,stdout_lines))
            if stderr_lines:
                self.logger.error("Client-{} \n STDERR {}".format(self.id,stderr_lines))
            self.ssh_client_handler.close()
            self.status = 0
        self.ssh_client_handler = None

def process_put_operation(master):
    """
    Manage the Upload process.

    Processes Stop Sequence:
    - Index-Monitor
    - Stop Workers if all index data distributed among client nodes.
    - Stop Status Poller
    - Stop Progress Tracer Monitor
    - Unmount remote NFS mounts.
    - Stop all clients
    :param master:
    :return:
    """
    workers_stopped = 0
    unmounted_nfs_shares = 0
    monitors_stopped = 0

    client_applications_termination_waiting_message = True # Used when Workers and Monitors are stopped, but ClientApps.
    while True:
        # Check for completion of indexing, Shutdown workers
        indexing_done = True

        if not master.indexing_started_flag.value:
            master.logger.info('Indexing on the shares not yet started')
            time.sleep(1)
            continue

        if len(master.progress_of_indexing) and not workers_stopped:
            indexing_done = False

        ## Check if index generation is completed by the worker processes.
        if indexing_done and master.index_data_generation_complete.value == 0:
            # Shut down Monitor-Index at Master
            index_generation_time = (datetime.now() - master.operation_start_time).seconds                       
            master.logger.info('Index generation completed, Time: {} sec'.format(index_generation_time))
            #print('INFO: Indexing the shares completed')
            master.index_data_lock.acquire()
            master.index_data_generation_complete.value = 1
            master.index_data_lock.release()

        # Check all the ClientApplications once they are finished
        all_clients_completed = 1
        for client in master.clients:
            #print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
            if not client.status:
                if client.remote_client_status() :
                    client.remote_client_exit_status()
                all_clients_completed = 0

        if all_clients_completed:
            master.logger.info("All ClientApplications terminated gracefully !") ## Termination3
        elif  workers_stopped and monitors_stopped:
            if client_applications_termination_waiting_message:
                master.logger.info("Waiting for ClientApplication to stop!")
                client_applications_termination_waiting_message = False

        # Check for Monitors status
        master.monitor.status_lock.acquire()
        if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
                master.monitor.monitor_status_poller.value and \
                master.monitor.monitor_progress_status.value:
            monitors_stopped = 1
            #print("INFO: All Monitors belongs to Master terminated!")
            master.logger.info("All monitors belongs to Master terminated!")
        master.monitor.status_lock.release()

        # Un-mount device once all monitors are stopped. Because, if ClientApp is launches at the same node of master, then
        # un-mount should not happen.
        if monitors_stopped and  not unmounted_nfs_shares:
            master.logger.info("Un-mount all NFS shares at Master")
            #master.nfs_cluster_obj.umount_all()  ## Termination2
            unmounted_nfs_shares = 1


        # Bring down workers.
        if not workers_stopped  and master.monitor.monitor_index_data_sender.value:
            master.stop_workers()
            workers_stopped = 1

        ## Once all the response received from Client Applications, shut down 3 monitors
        if workers_stopped and monitors_stopped and all_clients_completed:
            break

        # If ClientApplications running on client nodes gets terminated, then shutdown workers, monitors forcefully to exit program
        if all_clients_completed:
            if not monitors_stopped:
                master.stop_monitor()
            if not workers_stopped:
                master.stop_workers()
            break

        time.sleep(2)



def process_list_operation(master):
    """
    Perform LIST operation
    - Initiate LISTing
    - Check for completion of LIST operation
    - Wait for ObjectKeysAggregator to finish writing to a file
    - Shutdown workers
    :param master: Master object
    :return: None
    """
    #with master.listing_only.get_lock():
    master.listing_only.value = True
    while True:
      progress_bar("Operation LIST is in Progress")
      try:
        # Check for completion of listing
        master.listing_progress_lock.acquire()
        if master.listing_status.value == 1 and len(master.listing_progress) == 0:
          listing_time = (datetime.now() - master.operation_start_time).seconds
          master.logger.info("LISTing is completed in {} seconds".format(listing_time))
          master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))
          master.listing_status.value = 2
        master.listing_progress_lock.release()

        if master.listing_aggregation_status and master.listing_aggregation_status.value == 1:

          master.stop_workers()
          break
      except Exception as e:
        master.logger.excep("Listing - {}".format(e))
      time.sleep(1)

def process_del_operation(master):
    workers_stopped = 0
    monitors_stopped = 0

    while True:
        progress_bar("Operation DEL in Progress!")
        # Check for completion of listing, Shutdown workers
        listing_done = False
        master.listing_progress_lock.acquire()
        # Determine listing is done.
        if master.listing_status.value == 1 and len(master.listing_progress) == 0 :
            listing_done = True
        master.listing_progress_lock.release()

        if not workers_stopped:
            if listing_done and master.index_data_generation_complete.value == 0:
                master.index_data_generation_complete.value = 1
                #master.logger.info("Object-Keys generation through listing is completed!")
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info("LISTING Completed for {} operation in {} seconds".format(master.operation, listing_time))
                master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))
                # Shutdown workers
                #master.stop_workers()
                #workers_stopped = 1


        # Check all the ClientApplications once they are finished
        all_clients_completed = 1
        for client in master.clients:
            #print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
            if not client.status:
                if client.remote_client_status() :
                    client.remote_client_exit_status()
                all_clients_completed = 0

        # Check for Monitors status
        master.monitor.status_lock.acquire()
        if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
                master.monitor.monitor_status_poller.value and \
                master.monitor.monitor_progress_status.value:
            monitors_stopped = 1
        master.monitor.status_lock.release()

        # Bring down workers.
        if not workers_stopped and master.monitor.monitor_index_data_sender.value:
            master.stop_workers()
            workers_stopped = 1

        ## Once all the response received from Client Applications, shut down 3 monitors
        if workers_stopped and monitors_stopped and all_clients_completed:
            break

        # If ClientApplications running on client nodes gets terminated, then shutdown workers, monitors forcefully to exit program
        if all_clients_completed:
            if not monitors_stopped:
                master.stop_monitor()
            if not workers_stopped:
                master.stop_workers()
            break

        time.sleep(2)

def process_get_operation(master):
    workers_stopped = 0
    monitors_stopped = 0

    while True:
        progress_bar("Operation GET in progress!")
        # Check for completion of indexing, Shutdown workers
        listing_done = False
        master.listing_progress_lock.acquire()
        # Determine listing is done.
        if master.listing_status.value == 1 and len(master.listing_progress) == 0:
            listing_done = True
        master.listing_progress_lock.release()

        if not workers_stopped:
            if listing_done and master.index_data_generation_complete.value == 0:
                master.index_data_generation_complete.value = 1
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info("LISTING Completed for {} operation in {} seconds".format(master.operation, listing_time))
                master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))
        # Shutdown workers
        # master.stop_workers()
        # workers_stopped = 1

        # Check all the ClientApplications once they are finished
        all_clients_completed = 1
        for client in master.clients:
            # print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
            if not client.status:
                if client.remote_client_status():
                    client.remote_client_exit_status()
                all_clients_completed = 0

        # Check for Monitors status
        master.monitor.status_lock.acquire()
        if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
                master.monitor.monitor_status_poller.value and \
                master.monitor.monitor_progress_status.value:
            monitors_stopped = 1
        master.monitor.status_lock.release()

        # Bring down workers.
        if not workers_stopped and master.monitor.monitor_index_data_sender.value:
            master.stop_workers()
            workers_stopped = 1

        ## Once all the response received from Client Applications, shut down 3 monitors
        if workers_stopped and monitors_stopped and all_clients_completed:
            break

        # If ClientApplications running on client nodes gets terminated, then shutdown workers, monitors forcefully to exit program
        if all_clients_completed:
            if not monitors_stopped:
                master.stop_monitor()
            if not workers_stopped:
                master.stop_workers()
            break

        time.sleep(1)


if __name__ == "__main__":
    cli = CommandLineArgument()
    operation = cli.operation.upper()
    # Add signal handler
    #signal_handler = SignalHandler()
    #signal_handler.initiate()

    params = cli.options
    config_obj = Config(params)
    config = config_obj.get_config()
    #print(config)

    master = Master(operation, config)
    if config.get("debug", False):
        master.logging_level = "DEBUG"
    now = datetime.now()
    master.start()
    master.logger.info("DataMover Config Options : {}".format(config))
    master.logger.info("Started DataMover with Logger in {} mode!".format(master.logging_level))

    #signal_handler.registered_functions.append(master.nfs_cluster_obj.umount_all)

    if operation == "PUT":
        process_put_operation(master)
        master.nfs_cluster_obj.umount_all()
    elif operation == "LIST":
        process_list_operation(master)
    elif operation == "DEL":
        process_del_operation(master)
    elif operation == "GET":
        process_get_operation(master)
    elif operation == "TEST":
        if config["data_integrity"]:
            process_put_operation(master)
            master.nfs_cluster_obj.umount_all()
            if master.testcase_passed.value:
                master.logger.info("###### DataIntegrity Test Status: PASSED ######")
            else:
                master.logger.error("###### DataIntegrity Test Status: FAILED ######")

    # Start Compaction
    if "compaction" in params and params["compaction"]:
        master.compaction()

    # Terminate logger at the end.
    master.stop_logging()  ## Termination5
    print("INFO: Stopping master")
    ### 10.1.51.238 , 10.1.51.54, 10.1.51.61



