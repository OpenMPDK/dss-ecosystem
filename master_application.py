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
import os, sys
from utils.utility import exception, exec_cmd, remoteExecution, get_s3_prefix, progress_bar, get_ip_address
from utils.utility import is_prefix_valid_for_nfs_share, validate_s3_prefix
from utils.config import Config, commandLineArgumentParser, CommandLineArgument
from utils.signal_handler import SignalHandler
from utils import __VERSION__
from logger import MultiprocessingLogger

from multiprocessing import Process, Queue, Value, Lock, Manager
from worker import Worker
from monitor import Monitor
from nfs_cluster import NFSCluster
from task import Task, iterate_dir
import time
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
        self.client_ip_list = config.get("tess_clients_hosts_or_ip_addresses", [])
        self.workers_count = config["master"]["workers"]
        self.max_index_size = config["master"]["max_index_size"]
        self.index_data_queue_size = config["master"].get("index_data_queue_size", 10000)
        self.workers = []
        self.clients = []
        self.task_queue = Queue()
        self.task_lock = Lock()
        self.dryrun = config.get("dryrun", False)

        self.index_data_json_file = '/var/log/prefix_index_data.json'

        if "environment" in config:
            self.dir_path = config["environment"].get("target_dir", "/usr/dss/nkv-datamover")
        else:
            self.dir_path = "/usr/dss/nkv-datamover"

        self.operation = operation
        self.client_user_id = config["client"]["user_id"]
        self.client_password = config["client"]["password"]

        self.s3_config = config.get("s3_storage", {})
        self.nfs_config = self.config.get("nfs_config", {})

        self.standalone = config.get("standalone", False)

        # Logging
        self.logging_path = "/var/log/dss"
        self.logging_level = "INFO"
        if "logging" in config:
            self.logging_path = config["logging"].get("path", "/var/log/dss")
            self.logging_level = config["logging"].get("level", "INFO")
        self.logger = None
        self.logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
        self.logger_queue = Queue()
        self.logger_lock = Lock()

        self.lock = Lock()

        # Operation PUT/GET/DEL/LIST
        self.index_data_queue = Queue()  # Index data stored by workers
        self.index_data_lock = Lock()
        self.index_data_generation_complete = Value('i', 0)  ##TODO - Set value when all indexing is done.
        self.indexing_started_flag = Value('i', 0) # [0,1,2,-1] => ["READY", "STARTED", "COMPLETED", "FAILED"]

        # Operation LIST
        self.prefix = config.get("prefix", None)
        self.listing_progress = Value('i', 0)
        self.listing_status = Value('i', 0) # [0,1,2] => ['NOT STARTED', 'STARTED', 'COMPLETED']
        self.listing_only = Value('b', False)
        self.listing_aggregation_status = Value('i', 0)
        self.listing_objectkey_queue = Queue()

        # Status Progress
        self.index_data_count = Value('i', 0)  # File Index count shared between worker process.
        self.index_msg_count = Value('i', 0)  # How many messages have been produced by the producers.
        self.received_index_msg_count = Value('i', 0)  # How many messages have been consumed by the consumers.
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

        # IP Address Family
        self.ip_address_family = config.get("ip_address_family", "IPV4")

        # Unit TestCase
        self.testcase_passed = Value('b', False)

        # Status queue
        if self.standalone:
            self.operation_status_queue = Queue()
        else:
            self.operation_status_queue = None

        self.prefix_index_data = manager.dict()

        try:
            if os.path.exists(self.index_data_json_file):
                print("Reading existing index-meta-data - {}".format(time.time()))
                with open(self.index_data_json_file) as f:
                    index_data = json.load(f)
                    print("Loaded the index-meta-data - {}".format(time.time()))
                    self.prefix_index_data.update(index_data)
                    # print("Loaded the index data to manager dict - {}".format(time.time()))
        except Exception as e:
            print("Exception in loading prefix_index_data.json file", e)

    def __del__(self):
        # Stop workers
        # self.stop_workers()
        # Stop Monitor
        # Stop Messaging
        # Stop clients
        # self.stop_logging()
        pass
        # Remove local NFS share

    def start(self):
        """
        Start the master application with the following stuff
        - Start logger
        - Launch workers - Proceed only when at least one worker is up.
        - Spawn clients
        - Start Monitor
        - Start Messaging ( File indexing message channel, progress status )
        :return:
        """
        self.start_logging()
        self.logger.info("Performing {} operation".format(self.operation))
        if not self.start_workers():
            self.logger.info("Exit DataMover!")
            self.stop_logging()
            sys.exit("Workers were not started. Shutting down DataMover application")

        self.operation_start_time = datetime.now()
        if not self.standalone:
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
                       status_queue=self.operation_status_queue,
                       progress_of_indexing=self.progress_of_indexing,
                       progress_of_indexing_lock=self.progress_of_indexing_lock,
                       index_data_count=self.index_data_count,
                       index_msg_count=self.index_msg_count,
                       listing_progress=self.listing_progress,
                       s3_config=self.s3_config,
                       indexing_started_flag=self.indexing_started_flag,
                       listing_status=self.listing_status,
                       listing_only=self.listing_only,
                       listing_objectkey_queue=self.listing_objectkey_queue,
                       index_data_queue_size=self.index_data_queue_size,
                       prefix_index_data=self.prefix_index_data,
                       standalone=self.standalone
                       )
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
                # self.logger.info("DEBUG: worker-{}, status-{}".format(w.id, w.status.value))
            if s3_connection_failed or workers_started:
                break 
                 
        if not workers_started:
            self.logger.fatal("Workers were not started exit application!")
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
        self.logger.info("Stopped all the workers running with MasterApplication! ")

    def spawn_clients(self):
        """
        Spawn the clients
        :return:
        """
        index = 0
        for client_ip in self.client_ip_list:
            client = Client(index,
                            client_ip,
                            self.operation,
                            self.logger,
                            self.config)
            client.start()
            self.logger.info("Started ClientApplication-{} at node - {}".format(client.id, client_ip))
            self.clients.append(client)
            index += 1

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
                               index_data_json_file=self.index_data_json_file,
                               index_msg_count=self.index_msg_count,
                               received_index_msg_count=self.received_index_msg_count,
                               logger=self.logger,
                               operation=self.operation,
                               operation_start_time=self.operation_start_time,
                               listing_status=self.listing_status,
                               listing_aggregation_status=self.listing_aggregation_status,
                               listing_objectkey_queue=self.listing_objectkey_queue,
                               testcase=self.testcase_passed,
                               prefix_index_data=self.prefix_index_data,
                               status_queue=self.operation_status_queue,
                               standalone=self.standalone
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
        self.logger.info("** DataMover VERSION:{} **".format(__VERSION__))
        self.logger.info("Started Logger with {} mode!".format(self.logging_level))

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
        # Validate S3 prefix
        if self.prefix and not validate_s3_prefix(self.logger, self.prefix):
            self.logger.fatal("Bad prefix specified, exit application.")
            self.indexing_started_flag.value = -1
            return

        # First Mount all NFS share locally
        self.nfs_cluster_obj = NFSCluster(self.config.get("nfs_config", {}), "root", "", self.logger)
        self.nfs_cluster_obj.mount_all()
        if self.nfs_cluster_obj.mounted == False:
            self.logger.fatal("Mounting failed, EXIT indexing!")
            self.indexing_started_flag.value = -1
            return

        # Create first level task for each NFS share
        local_mounts = self.nfs_cluster_obj.get_mounts()
        for ip_address, nfs_shares in local_mounts.items():
            self.logger.info("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
            self.nfs_shares.extend(nfs_shares)
            for nfs_share in nfs_shares:
                if self.prefix:
                    if is_prefix_valid_for_nfs_share(self.logger, share=nfs_share, ip_address=ip_address,
                                                     prefix=self.prefix):
                        nfs_share_mount = os.path.abspath("/" + self.prefix)
                    else:
                        continue
                else:
                    # print("DEBUG: Creating task for {}".format(nfs_share))
                    nfs_share_mount = os.path.abspath("/" + ip_address + "/" + nfs_share)
                task = Task(operation="indexing",
                            data=nfs_share_mount,
                            nfs_cluster=ip_address,
                            nfs_share=nfs_share,
                            prefix=self.prefix,
                            max_index_size=self.max_index_size,
                            s3config=self.config["s3_storage"],
                            dryrun=self.dryrun
                            )
                self.task_queue.put(task)

    def start_listing(self):
        self.logger.info("Started Listing operation!")
        bad_prefix_no_listing = True
        if self.operation.upper() == "LIST":
            self.listing_only.value = True

        prefix_index_data = {}
        listing_based_on_indexing = False
        if os.path.exists(self.index_data_json_file):
            listing_based_on_indexing = True
            with open(self.index_data_json_file, "r") as prefix_index_data_handler:
                try:
                    prefix_index_data = json.load(prefix_index_data_handler)
                except json.JSONDecodeError as e:
                    self.logger.error("Persistent index data - {}".format(e))
                except MemoryError as e:
                    self.logger.error("Unable to load prefix_index_data - {}".format(e))

        if prefix_index_data:
            self.logger.info("Using {} file for LISTing".format(self.index_data_json_file))
            for prefix in prefix_index_data.keys():
                if self.prefix and not prefix.startswith(self.prefix):
                    continue
                bad_prefix_no_listing = False
                with self.listing_progress.get_lock():
                    self.listing_progress.value +=1
                task = Task(operation="list",
                            data={"prefix": prefix},
                            s3config=self.config["s3_storage"],
                            max_index_size=self.config["master"].get("max_index_size", 10),
                            listing_based_on_indexing=listing_based_on_indexing
                            )
                self.task_queue.put(task)
        else:
            for prefix in get_s3_prefix(self.logger, self.config.get("nfs_config", {}), self.prefix):
                bad_prefix_no_listing = False
                with self.listing_progress.get_lock():
                    self.listing_progress.value += 1
                task = Task(operation="list",
                            data={"prefix": prefix},
                            s3config=self.config["s3_storage"],
                            max_index_size=self.config["master"].get("max_index_size", 10),
                            listing_based_on_indexing=listing_based_on_indexing
                            )
                self.task_queue.put(task)
        if bad_prefix_no_listing:
            self.logger.error("LISTING failure!")
            self.listing_status.value = 1

    def compaction(self):
        # Spawn the process on target node and wait for response.
        command = "python3 {}/target_compaction.py".format(self.dir_path)
        compaction_status = {}
        start_time = datetime.now()
        for client_ip in self.config["dss_targets"]:
            self.logger.info("Started Compaction for target-ip:{}".format(client_ip))
            ssh_client_handler, stdin, stdout, stderr = remoteExecution(client_ip, self.client_user_id,
                                                                        self.client_password, command)
            compaction_status[client_ip] = {"status": False, "ssh_remote_client": ssh_client_handler, "stdout": stdout,
                                            "stderr": stderr}

        while True:
            is_compaction_done = True
            progress_bar("Compaction in Progress")
            for client_ip in compaction_status:
                if "status" in compaction_status[client_ip] and compaction_status[client_ip]["status"]:
                    continue
                if "ssh_remote_client" in compaction_status[client_ip] and compaction_status[client_ip][
                    "ssh_remote_client"]:
                    if "stdout" in compaction_status[client_ip] and compaction_status[client_ip]["stdout"]:
                        status = compaction_status[client_ip]["stdout"].channel.exit_status_ready()
                        if status:
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

    def __init__(self, id, host_or_ip_address, operation, logger, config):
        """
        Client object initiation
        :param id: A id for each ClientApplication
        :param ip: IP address of the node in which ClienApplication will run
        :param operation: PUT|GET|DEL|LIST
        :param logger: A Logger with shared Queue to be used by multiple process/workers.
        :param config: configuration.
        """
        self.id = id
        self.host_or_ip_address = host_or_ip_address
        self.ip_address = get_ip_address(logger, host_or_ip_address, config["ip_address_family"])
        self.operation = operation
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
        self.master_host_or_ip_address = config["master"]["host_or_ip_address"]
        self.dryrun = config.get("dryrun", False)
        self.destination_path = config.get("dest_path", "")  # Only to be used for GET operation

        self.env_gcc_source = None
        self.env_gcc_required = True
        if "environment" in config and "gcc" in config["environment"]:
            self.env_gcc_required = config["environment"]["gcc"].get("required", True)
            if self.env_gcc_required:
                self.env_gcc_source = config["environment"]["gcc"].get("source", "/usr/local/bin/setenv-for-gcc510.sh")
                self.logger.debug("Sourcing GCC environment from {} for GCC v-{}".format(
                    self.env_gcc_source, config["environment"]["gcc"].get("version", "GCC-VERSION-NOT-SPECIFIED")))

            self.dir_path = config["environment"].get("target_dir", "/usr/dss/nkv-datamover")
        else:
            self.dir_path = "/usr/dss/nkv-datamover"

        # Messaging service configuration
        if "message" in config:
            self.port_index = config["message"].get("port_index", "6000")  # Need to configure from configuration file.
            self.port_status = config["message"].get("port_status", "6001")
        else:
            self.port_index = "6000"
            self.port_status = "6001"
        self.socket_index = None
        self.socket_status = None

        # Remote output
        self.remote_execution_command = None
        self.remote_stdin = None
        self.remote_stdout = None
        self.remote_stderr = None

        # Execution status
        self.status = Value('b', False)
        self.exit_status_code = Value('i', -1)  # Integer value

    def __del__(self):
        """
        Receive the PID of the remote process, so that forcefully remote process can be terminated.
        :return:
        """
        if not self.status.value:
            self.stop()

    def start(self):
        """
        Remote execution of client application ("client_application.py")
        :return:
        """
        command = "python3  {}/client_application.py ".format(self.dir_path) + \
                  " --client_id {} ".format(self.id) + \
                  " --operation {} ".format(self.operation) + \
                  " --ip_address {} ".format(self.ip_address) + \
                  " --port_index {} ".format(self.port_index) + \
                  " --port_status {}  ".format(self.port_status)

        if self.operation.upper() == "GET" or self.operation.upper() == "TEST":
            command += " --dest_path {} ".format(self.destination_path)
        if self.operation.upper() == "TEST" and self.config.get("skip_upload", False):
            command += " --skip_upload "
        if self.master_host_or_ip_address == self.host_or_ip_address:
            command += " --master_node "
        if self.config.get("config", False):
            command += " --config {} ".format(self.config["config"])
        if self.dryrun:
            command += " --dryrun "
        if self.debug:
            command += " --debug "

        if self.env_gcc_required and self.env_gcc_source:
            command = "sh -c \" source {} && {} \"".format(self.env_gcc_source, command)
        self.ssh_client_handler, stdin, stdout, stderr = remoteExecution(self.ip_address, self.username, self.password,
                                                                         command)
        self.remote_stdin = stdin
        self.remote_stdout = stdout
        self.remote_stderr = stderr

    def remote_client_status(self):
        """
        It checks remote application and on its completion return True else False
        :return: Remote Process Completed (True) , Else (False)
        """
        if self.ssh_client_handler:
            if self.remote_stdout:
                self.status.value = self.remote_stdout.channel.exit_status_ready()
                if self.status.value:
                    self.logger.info("Remote ClientApplication-{} terminated!".format(self.id))
                return self.status.value
        return False

    def remote_client_exit_status(self):
        """
        It is blocking call.
        Check remote exit status code and close the socket connection.
        :return: exit status integer value, 0=Good, Non Zero value failure
        """
        if self.ssh_client_handler:

            self.exit_status_code.value = self.remote_stdout.channel.recv_exit_status()
            stdout_lines = self.remote_stdout.readlines()
            stderr_lines = self.remote_stderr.readlines()
            self.logger.debug("Client-{} Remote execution status: {}".format(self.id, self.exit_status_code.value))
            if self.exit_status_code.value:
                self.logger.warn("Client-{} \n STDOUT {}".format(self.id, stdout_lines))
                if stderr_lines:
                    self.logger.error("Client-{} \n STDERR {}".format(self.id, stderr_lines))
            self.ssh_client_handler.close()
            self.ssh_client_handler = None

    def stop(self):
        """
        Send a message to the client node through ZMQ.
        Need to terminate forcefully.
        :return:
        """
        self.logger.info("Stopping client-{}".format(self.id))
        if self.ssh_client_handler:
            self.exit_status_code.value = self.remote_stdout.channel.recv_exit_status()
            stdout_lines = self.remote_stdout.readlines()
            stderr_lines = self.remote_stderr.readlines()
            self.logger.info("Client-{} Remote execution status-{}".format(self.id, self.exit_status_code.value))
            if self.exit_status_code.value:
                self.logger.debug("Client-{} \n STDOUT {}".format(self.id, stdout_lines))
            if stderr_lines:
                self.logger.error("Client-{} \n STDERR {}".format(self.id, stderr_lines))
            self.ssh_client_handler.close()
            self.status.value = True
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
    all_clients_completed = 0
    all_client_terminated_gracefully = True

    client_applications_termination_waiting_message = True  # Used when Workers and Monitors are stopped, but ClientApps.
    while True:
        # Check for completion of indexing, Shutdown workers
        indexing_done = True
        # progress_bar("Object Count: {} ,Producer Msg Count: {}, Consumer Msg Count: {},  MSG-Queue Size - {}".format(master.index_data_count.value, master.index_msg_count.value, master.received_index_msg_count.value, master.index_data_queue.qsize()) )
        if master.indexing_started_flag.value == 0:
            time.sleep(1)
            continue
        if master.indexing_started_flag.value == -1:
            master.logger.fatal("EXIT PUT operation!")
            master.stop()
            break

        if len(master.progress_of_indexing) and not workers_stopped:
            indexing_done = False

        # Check if index generation is completed by the worker processes.
        if indexing_done and master.index_data_generation_complete.value == 0:
            # Shut down Monitor-Index at Master
            index_generation_time = (datetime.now() - master.operation_start_time).seconds
            master.logger.info('Index generation completed, Time: {} sec'.format(index_generation_time))
            master.logger.info("Indexed Object-Keys Count: {}".format(master.index_data_count.value))
            master.logger.info("Indexed Msg Count: {}".format(master.index_msg_count.value))
            master.index_data_lock.acquire()
            master.index_data_generation_complete.value = 1
            master.index_data_lock.release()

        if not master.standalone:
            if all_clients_completed == 0:
                completed = True  # Client Completion Status

                for client in master.clients:
                    if not client.status.value:
                        if client.remote_client_status():
                            client.remote_client_exit_status()
                            if client.exit_status_code.value != 0:
                                all_client_terminated_gracefully = False
                        else:
                            completed = False
                if completed:
                    if all_client_terminated_gracefully:
                        master.logger.info("All ClientApplications terminated gracefully !")
                    all_clients_completed = 1

            if workers_stopped and monitors_stopped:
                if client_applications_termination_waiting_message:
                    master.logger.info("Waiting for ClientApplication to stop!")
                    client_applications_termination_waiting_message = False

            # Check for Monitors status
            master.monitor.status_lock.acquire()
            if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
                    master.monitor.monitor_status_poller.value and \
                    master.monitor.monitor_progress_status.value:
                monitors_stopped = 1
                master.logger.info("All monitors belongs to Master terminated!")
            master.monitor.status_lock.release()

        if master.standalone:
            all_clients_completed = 1
            if not monitors_stopped and master.monitor.monitor_progress_status.value:
                monitors_stopped = 1
                master.logger.info('Monitor progress is completed')
                master.logger.info('Stopping all the workers')
                master.stop_workers()
                workers_stopped = 1
                master.logger.info('Stopped all the workers')

        # Un-mount device once all monitors are stopped. Because, if ClientApp is launches at the same node of master, then
        # un-mount should not happen.
        if monitors_stopped and not unmounted_nfs_shares:
            master.logger.info("Un-mount all NFS shares at Master")
            # master.nfs_cluster_obj.umount_all()  ## Termination2
            unmounted_nfs_shares = 1

        # Bring down workers.
        if not master.standalone and not workers_stopped and master.monitor.monitor_index_data_sender.value:
            master.stop_workers()
            workers_stopped = 1

        ## Once all the response received from Client Applications, shut down 3 monitors
        if workers_stopped and monitors_stopped and all_clients_completed:
            break

        # ERROR Handling
        if all_clients_completed and monitors_stopped and not workers_stopped:
            master.stop_workers()
            workers_stopped = 1
        # Shutdown client application if workers and monitor finished.
        # if workers_stopped and monitors_stopped and not all_clients_completed:
        #    for client in master.clients:
        #        if not client.status:
        #            if not client.remote_client_status():
        #                client.stop()
        #    all_clients_completed = 1

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
    while True:
        progress_bar("Listed Object Keys - {}".format(master.index_data_count.value))
        try:
            # Check for completion of listing
            #master.logger.info("Listing Status:{} , Listing Progress:{}".format(master.listing_status.value, master.listing_progress1.value))
            if master.listing_status.value == 1 and master.listing_progress.value == 0:
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info("LISTing is completed in {} seconds".format(listing_time))
                master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))
                master.listing_status.value = 2

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
        # Determine listing is done.
        if master.listing_status.value == 1 and master.listing_progress.value == 0:
            listing_done = True

        if not workers_stopped:
            if listing_done and master.index_data_generation_complete.value == 0:
                master.index_data_generation_complete.value = 1
                # master.logger.info("Object-Keys generation through listing is completed!")
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info(
                    "LISTING Completed for {} operation in {} seconds".format(master.operation, listing_time))
                master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))
                # Shutdown workers
                # master.stop_workers()
                # workers_stopped = 1

        # Check all the ClientApplications once they are finished
        all_clients_completed = 1
        for client in master.clients:
            # print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
            if not client.status.value:
                if client.remote_client_status():
                    client.remote_client_exit_status()
                else:
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
        # if all_clients_completed:
        #    if not monitors_stopped:
        #        master.stop_monitor()
        #    if not workers_stopped:
        #        master.stop_workers()
        #    break

        time.sleep(2)


def process_get_operation(master):
    workers_stopped = 0
    monitors_stopped = 0

    while True:
        progress_bar("Operation GET in progress!")
        # Check for completion of indexing, Shutdown workers
        listing_done = False
        # Determine listing is done.
        if master.listing_status.value == 1 and master.listing_progress.value == 0:
            listing_done = True

        if not workers_stopped:
            if listing_done and master.index_data_generation_complete.value == 0:
                master.index_data_generation_complete.value = 1
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info(
                    "LISTING Completed for {} operation in {} seconds".format(master.operation, listing_time))
                master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))
        # Shutdown workers
        # master.stop_workers()
        # workers_stopped = 1

        # Check all the ClientApplications once they are finished
        all_clients_completed = 1
        for client in master.clients:
            # print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
            if not client.status.value:
                if client.remote_client_status():
                    client.remote_client_exit_status()
                else:
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
        # if all_clients_completed:
        #    if not monitors_stopped:
        #        master.stop_monitor()
        #    if not workers_stopped:
        #        master.stop_workers()
        #    break

        time.sleep(1)


if __name__ == "__main__":
    cli = CommandLineArgument()
    operation = cli.operation.upper()
    # Add signal handler
    # signal_handler = SignalHandler()
    # signal_handler.initiate()

    params = cli.options
    config_obj = Config(params)
    config = config_obj.get_config()

    master = Master(operation, config)
    if config.get("debug", False):
        master.logging_level = "DEBUG"
    now = datetime.now()
    master.start()
    master.logger.info("DataMover Config Options : {}".format(config))
    master.logger.info("Started DataMover with Logger in {} mode!".format(master.logging_level))
    master.logger.info("IP Address Family - {}".format(master.ip_address_family))

    # signal_handler.registered_functions.append(master.nfs_cluster_obj.umount_all)

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


