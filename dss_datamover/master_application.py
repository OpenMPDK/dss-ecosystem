#!/usr/bin/

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

import json
import prctl
import os
import sys
import time

from utils.utility import (
    is_prefix_valid_for_nfs_share,
    get_s3_prefix,
    validate_s3_prefix,
    remoteExecution,
    progress_bar,
    get_ip_address
)
from utils.config import Config, CommandLineArgument
from utils.signal_handler import SignalHandler
from utils import __VERSION__
from logger import MultiprocessingLogger

from multiprocessing import Queue, Value, Lock, Manager, Event
from worker import Worker
from monitor import Monitor
from nfs_cluster import NFSCluster
from task import Task
from datetime import datetime


class Master(object):
    """
    Master application initiate
    -
    """

    def __init__(self, operation, config):
        manager = Manager()
        self.config = config
        self.client_ip_list = config.get("clients_hosts_or_ip_addresses", [])
        self.workers_count = config["master"]["workers"]
        self.max_index_size = config["master"]["max_index_size"]
        self.index_data_queue_size = config["master"].get("index_data_queue_size", 100000)
        self.workers = []
        self.clients = []
        self.task_queue = Queue()
        self.task_lock = Lock()
        self.dryrun = config.get("dryrun", False)

        self.process_monitor_event = Event()

        if "environment" in config:
            self.dir_path = config["environment"].get("target_dir", "/usr/dss/nkv-datamover")
        else:
            self.dir_path = "/usr/dss/nkv-datamover"

        self.operation = operation
        self.client_user_id = config["client"]["user_id"]
        self.client_password = config["client"]["password"]

        self.s3_config = config.get("s3_storage", {})

        self.fs_config = self.config.get("fs_config", {})
        self.server_as_prefix = self.fs_config.get("server_as_prefix", True)
        # override NFS configs with CLI args if applicable
        if {'nfs_server', 'nfs_share'}.issubset(self.config):
            if self.config['nfs_server'] and self.config['nfs_share']:
                self.fs_config['nfs'] = {self.config['nfs_server']: [self.config['nfs_share']]}
        if 'nfs_port' in self.config:
            self.fs_config['nfsport'] = self.config['nfs_port']
        # pass through server-as-prefix option as part of NFS configs
        # self.fs_config['server_as_prefix'] = self.config['server_as_prefix']

        self.standalone = config.get("standalone", False)

        # Logging
        if "logging" in config:
            self.logging_path = config["logging"].get("path", "/var/log/dss")
            self.logging_level = config["logging"].get("level", "INFO")
            self.max_log_file_size = config["logging"].get("max_log_file_size", (1024 * 1024))
            self.log_file_backup_count = config["logging"].get("log_file_backup_count", 5)
        else:
            self.logging_path = "/var/log/dss"
            self.logging_level = "INFO"
            self.max_log_file_size = (1024 * 1024)
            self.log_file_backup_count = 5

        self.logger = None
        self.logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
        self.logger_queue = Queue()

        self.lock = Lock()
        # Metadata files
        self.index_data_json_file = '{}/prefix_index_data.json'.format(self.logging_path)
        self.resume_prefix_dir_keys_file = '{}/dm_resume_prefix_dir_keys.txt'.format(self.logging_path)

        # Operation PUT/GET/DEL/LIST
        self.index_data_queue = Queue()  # Index data stored by workers
        self.index_data_lock = Lock()
        self.index_data_generation_complete = Value('i', 0)  # TODO - Set value when all indexing is done.
        self.indexing_started_flag = Value('i', 0)  # [0,1,2,-1] => ["READY", "STARTED", "COMPLETED", "FAILED"]

        # Operation LIST
        self.prefix = config.get("prefix", None)
        self.prefixes = []  # TODO need to take multiple prefix from command line and store into list.
        self.listing_progress = Value('i', 0)
        self.listing_status = Value('i', 0)  # [0,1,2] => ['NOT STARTED', 'STARTED', 'COMPLETED']
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

        # Unit TestCase
        self.testcase_passed = Value('b', False)

        # Status queue
        if self.standalone:
            self.operation_status_queue = Queue()
        else:
            self.operation_status_queue = None

        self.prefix_index_data = manager.dict()

        # Get the directory prefix keys that are yet to be resumed for PUT operation
        self.dir_prefixes_to_resume = list()
        self.resume_flag = False

    def __del_dummy__(self):
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
        self.load_prefix_index_data()  # Load prefix metadata.
        self.load_prefix_keys_for_resume_operation()  # Check if resume operation required.

        # Validate S3 prefix before starting the workers, to allow graceful exit of application for bad prefix
        # (Fix for MIN-1312)
        if self.prefix:
            if not validate_s3_prefix(self.logger, self.prefix, self.fs_config.get('nfs', {}), server_as_prefix=self.server_as_prefix):
                self.stop_logging()
                sys.exit("Invalid prefix. Shutting down DataMover application")

        if not self.start_workers():
            self.logger.info("Exit DataMover!")
            self.stop_logging()
            sys.exit("Workers were not started. Shutting down DataMover application")
        self.operation_start_time = datetime.now()

        if not self.standalone:
            if not (self.operation.upper() == "LIST" and not self.config.get("distributed", False)):
                self.stop_client_stale_process_on_clients()
                self.spawn_clients()

        if not self.process_monitor_event.is_set() and self.operation.upper() in ["PUT", "TEST"]:
            self.start_indexing()
        if not self.process_monitor_event.is_set() and self.operation.upper() in ["LIST", "DEL", "GET"]:
            self.start_listing()

        try:
            if not self.process_monitor_event.is_set():
                self.start_monitor()
        except Exception as e:
            self.logger.excep("Exception in starting monitor {}".format(e))

        self.logger.info("DataMover running with \"{}\"  S3 client".format(self.s3_config["client_lib"]))

    def stop(self, force_flag=False):
        """
        Stop master and its all clients and their application.
        - Send notification to clients to stop
        - Stop all the workers
        - Stop Monitor
        - Stop messaging systems.
        :return:
        """
        self.stop_workers()
        self.stop_monitor()
        self.stop_clients(force_flag=force_flag)
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
            try:
                w.start()
                self.workers.append(w)
            except:
                pass
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

    def stop_workers(self, worker_id=None):
        """
        Stop worker/s
        :param worker_id: worker id
        :return:
        """
        index = 0
        for w in self.workers:
            if worker_id and w.id != worker_id:
                continue
            if w.get_status():
                w.stop()
            index += 1
        self.logger.info("Stopped all the workers running with MasterApplication! ")

    def stop_client_stale_process_on_clients(self):
        """
        Spawn the clients
        :return:
        """
        index = 0
        for client_ip in self.client_ip_list:
            client = Client(
                index,
                client_ip,
                self.operation,
                self.logger,
                self.config,
                self.server_as_prefix
            )
            try:
                self.logger.info(f"Stopping stale ClientApplication on node {client_ip}")
                client.stop_process_on_remote_client()
                self.logger.info(f"Stopped stale ClientApplication on node {client_ip}")
            except Exception as e:
                self.logger.excep(f"Stopping stale ClientApplication on node {client_ip} with exception {e}")

    def spawn_clients(self):
        """
        Spawn the clients
        :return:
        """
        index = 0
        for client_ip in self.client_ip_list:
            client = Client(
                index,
                client_ip,
                self.operation,
                self.logger,
                self.config,
                self.server_as_prefix
            )
            try:
                client.start()
                self.logger.info("Started ClientApplication-{} at node - {}".format(client.id, client_ip))
                self.clients.append(client)
            except Exception as e:
                self.logger.error(f'Error in starting client application on node {client_ip} - {e}')
            index += 1

    def stop_clients(self, force_flag=False):
        """
        Stop all the client application associated with this master application
        :return:
        """
        for client in self.clients:
            client.stop(force_flag=force_flag)

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
                               resume_prefix_dir_keys_file=self.resume_prefix_dir_keys_file,
                               status_queue=self.operation_status_queue,
                               standalone=self.standalone,
                               resume_flag=self.resume_flag,
                               dryrun=self.dryrun
                               )
        self.monitor.start()

    def set_proc_monitor_event(self):
        try:
            self.process_monitor_event.set()
            self.logger.info("Process monitor event is set to stop all the processes")
        except Exception as e:
            self.logger.excep(f"Exception in setting proc monitor event - {e}")
            raise

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
        self.logger = MultiprocessingLogger(self.logger_queue, self.logger_status, self.max_log_file_size, self.log_file_backup_count)
        self.logger.config(self.logging_path,
                           __file__,
                           self.logging_level)
        self.logger.start()
        self.logger.create_logger_handle()
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
        if self.prefix:
            if validate_s3_prefix(self.logger, self.prefix, server_as_prefix=self.server_as_prefix):
                self.prefixes = [self.prefix]
            else:
                self.logger.fatal("Bad prefix specified, exit application.")
                self.indexing_started_flag.value = -1
                return

        # Create NFS cluster object
        self.nfs_cluster_obj = NFSCluster(self.fs_config, "root", "", self.logger)

        if self.resume_flag:
            self.prefixes = self.dir_prefixes_to_resume

        if self.prefixes:
            for prefix in self.prefixes:
                if self.process_monitor_event.is_set():
                    self.logger.info('start_indexing: Processing prefixes for indexing stopped')
                    break
                self.logger.info("Processing prefix:{}".format(prefix))
                (ip_address, nfs_share, ret, out) = self.nfs_cluster_obj.mount_based_on_prefix(prefix)
                if ret != 0:
                    self.nfs_cluster_obj.umount_all()
                    self.logger.error("NFS Mounting failed, Error: {}".format(out))
                    self.logger.fatal("Mounting failed, EXIT indexing!")
                    self.indexing_started_flag.value = -1
                    return
                if ret == 0 and is_prefix_valid_for_nfs_share(self.logger, share=nfs_share, ip_address=ip_address,
                                                              prefix=prefix, server_as_prefix=self.server_as_prefix):
                    nfs_share_prefix_path = os.path.abspath("/" + prefix)
                    task = Task(operation="indexing",
                                data=nfs_share_prefix_path,
                                nfs_cluster=ip_address,
                                nfs_share=nfs_share,
                                max_index_size=self.max_index_size,
                                s3config=self.config["s3_storage"],
                                dryrun=self.dryrun,
                                resume_flag=self.resume_flag
                                )
                    self.task_queue.put(task)
        else:
            self.nfs_cluster_obj.mount_all()
            if not self.nfs_cluster_obj.local_mounts:
                self.logger.fatal("Mounting failed, EXIT indexing!")
                self.indexing_started_flag.value = -1
                return

            # Create first level task for each NFS share
            local_mounts = self.nfs_cluster_obj.get_mounts()
            for ip_address, nfs_shares in local_mounts.items():
                if self.process_monitor_event.is_set():
                    self.logger.info('start_indexing: Processing NFS Shares for indexing stopped')
                    break
                self.logger.info("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
                self.nfs_shares.extend(nfs_shares)
                for nfs_share in nfs_shares:
                    # print("DEBUG: Creating task for {}".format(nfs_share))
                    if self.nfs_cluster_obj.mounted:
                        nfs_share_mount = nfs_share
                    else:
                        nfs_share_mount = os.path.abspath("/" + ip_address + "/" + nfs_share) if self.server_as_prefix else os.path.abspath("/" + nfs_share)
                    task = Task(operation="indexing",
                                data=nfs_share_mount,
                                nfs_cluster=ip_address,
                                nfs_share=nfs_share,
                                prefix=self.prefix,
                                max_index_size=self.max_index_size,
                                s3config=self.config["s3_storage"],
                                dryrun=self.dryrun,
                                resume_flag=self.resume_flag
                                )
                    self.task_queue.put(task)

    def start_listing(self):
        self.logger.info("Started Listing operation!")
        bad_prefix_no_listing = True
        dump_object_keys_path = None
        listing_based_on_indexing = False
        if self.operation.upper() == "LIST":
            self.listing_only.value = True
            dump_object_keys_path = self.config.get("dest_path", None)

        # Distributed LISTing
        # TODO distributed prefix_dir from dm_resume_* file.
        distributed_listing = self.config.get("distributed", False)
        if distributed_listing:
            self.logger.info("Performing distributed LISTing!")
            for prefix in self.prefix_dirs:
                bad_prefix_no_listing = False
                if self.prefix and not prefix.startswith(self.prefix):
                    continue
                message = {"dir": prefix}
                self.index_data_queue.put(message)
                with self.index_msg_count.get_lock():
                    self.index_msg_count.value += 1
            self.index_data_generation_complete.value = 1
        else:
            # Standalone LISTing on single node.
            self.logger.info("Standalone LISTing..")
            if self.prefix_index_data:
                self.logger.info("PREFIX INDEX DATA..")
                listing_based_on_indexing = True
                self.logger.info("Using {} file for LISTing".format(self.index_data_json_file))
                for prefix in self.prefix_index_data.keys():
                    if self.prefix and not prefix.startswith(self.prefix):
                        continue
                    bad_prefix_no_listing = False
                    with self.listing_progress.get_lock():
                        self.listing_progress.value += 1
                    task = Task(operation="list",
                                data={"prefix": prefix},
                                s3config=self.config["s3_storage"],
                                max_index_size=self.config["master"].get("max_index_size", 10),
                                listing_based_on_indexing=listing_based_on_indexing,
                                dest_path=dump_object_keys_path
                                )
                    self.task_queue.put(task)
            else:
                self.logger.info("NOT IN PREFIX INDEX DATA..")
                for prefix in get_s3_prefix(self.logger, self.fs_config.get("nfs", {}), self.prefix, server_as_prefix=self.server_as_prefix):
                    bad_prefix_no_listing = False
                    with self.listing_progress.get_lock():
                        self.listing_progress.value += 1
                    self.logger.info(f"^^^^^^^^^^^^^^^^^^^ PREFIX BEING PROCESSED IS {prefix}")
                    task = Task(operation="list",
                                data={"prefix": prefix},
                                s3config=self.config["s3_storage"],
                                max_index_size=self.config["master"].get("max_index_size", 10),
                                listing_based_on_indexing=listing_based_on_indexing,
                                dest_path=dump_object_keys_path
                                )
                    self.task_queue.put(task)
        if bad_prefix_no_listing:
            self.logger.error("LISTING failure!")
            self.listing_status.value = 1

    def compaction(self):
        """
        Initiate the compaction process into remote target node.
        :return:None
        """
        # Spawn the process on target node and wait for response.
        target_compaction_source = "python3 {}/target_compaction.py".format(self.dir_path)
        compaction_status = {}
        start_time = datetime.now()

        # Check if subsystem-nqn are specified
        if "dss_targets" not in self.config:
            self.logger.fatal("Target information are not specified!")
            return
        if "subsystem_nqn" not in self.config["dss_targets"]:
            self.logger.fatal("Target subsystem nqn information not specified!")
            return

        for target_ip in self.config["dss_targets"]["subsystem_nqn"]:
            command = target_compaction_source + " --ip_address " + target_ip
            command += " --logdir " + self.config["logging"]["path"]
            command += " --user_id " + self.client_user_id
            if self.client_password:
                command += " --password " + self.client_password

            subsystem_nqn_str = ",".join(self.config["dss_targets"]["subsystem_nqn"][target_ip])
            command += " --subsystem_nqn " + subsystem_nqn_str

            # Add target installation path
            if "installation_path" in self.config["dss_targets"]:
                command += " --installation_path " + self.config["dss_targets"]["installation_path"]

            self.logger.info("Started Compaction for target-ip:{}".format(target_ip))
            ssh_client_handler, stdin, stdout, stderr = remoteExecution(target_ip, self.client_user_id,
                                                                        self.client_password, command)
            compaction_status[target_ip] = {"status": False, "ssh_remote_client": ssh_client_handler, "stdout": stdout,
                                            "stderr": stderr}
        # Wait for target compaction to finish.
        while True:
            is_compaction_done = True
            progress_bar("Compaction in Progress")
            for target_ip in compaction_status:
                if "status" in compaction_status[target_ip] and compaction_status[target_ip]["status"]:
                    continue
                if ("ssh_remote_client" in compaction_status[target_ip]
                        and compaction_status[target_ip]["ssh_remote_client"]):
                    if "stdout" in compaction_status[target_ip] and compaction_status[target_ip]["stdout"]:
                        status = compaction_status[target_ip]["stdout"].channel.exit_status_ready()
                        if status:
                            self.logger.info("Compaction is finished for - {}".format(target_ip))
                            compaction_status[target_ip]["status"] = True
                            compaction_status[target_ip]["ssh_remote_client"].close()
                        else:
                            is_compaction_done = False
            if is_compaction_done:
                break

        compaction_time = (datetime.now() - start_time).seconds
        self.logger.info("Total Compaction time - {} seconds".format(compaction_time))

    def load_prefix_index_data(self):
        if self.operation.upper() == "LIST":
            if os.path.exists(self.resume_prefix_dir_keys_file):
                self.prefix_dirs = []
                with open(self.resume_prefix_dir_keys_file, "r") as FH:
                    lines = FH.read()
                    self.prefix_dirs = lines[:-1].split("\n")
        else:
            if os.path.exists(self.index_data_json_file):
                with open(self.index_data_json_file, "r") as prefix_index_data_handler:
                    try:
                        start_loading_existing_metadata = datetime.now()
                        self.prefix_index_data = json.load(prefix_index_data_handler)
                        self.logger.info("Loaded the {} - {} seconds".format(self.index_data_json_file,
                                                                             (datetime.now() - start_loading_existing_metadata).seconds))
                    except json.JSONDecodeError as e:
                        self.logger.error("JSONDecodeError - Persistent index data - {}".format(e))
                    except MemoryError as e:
                        self.logger.error("MemoryError - Unable to load prefix_index_data - {}".format(e))

    def load_prefix_keys_for_resume_operation(self):
        if self.operation.upper() == 'PUT':
            if self.prefix_index_data:
                try:
                    if os.path.exists(self.resume_prefix_dir_keys_file):
                        self.logger.info("Reading existing prefix dirs for DM resume - {}".format(time.time()))
                        with open(self.resume_prefix_dir_keys_file) as f:
                            lines = f.read()
                            self.logger.info("Loaded the prefix dirs for DM resume - {}".format(time.time()))
                            keys_already_uploaded = lines.split('\n')
                            self.dir_prefixes_to_resume = list(set(self.prefix_index_data.keys()) - set(keys_already_uploaded))
                            if self.dir_prefixes_to_resume:
                                self.logger.info("Datamover running in RESUME mode")
                                self.resume_flag = True
                            else:
                                self.logger.info("All the directories are up to date. Exiting")
                                self.stop_logging()
                                sys.exit(0)
                except Exception as e:
                    self.logger.excep("Exception in loading prefix dirs file for DM resume - {}", e)
            else:
                try:
                    os.unlink(self.resume_prefix_dir_keys_file)
                except FileNotFoundError:
                    pass


class Client(object):

    def __init__(self, id, host_or_ip_address, operation, logger, config, server_as_prefix):
        """
        Client object initiation
        :param id: A id for each ClientApplication
        :param host_or_ip_address: IP address of the node in which ClienApplication will run
        :param operation: PUT|GET|DEL|LIST
        :param logger: A Logger with shared Queue to be used by multiple process/workers.
        :param config: configuration.
        """
        self.id = id
        self.host_or_ip_address = host_or_ip_address
        self.ip_address = get_ip_address(logger, host_or_ip_address)
        self.operation = operation
        self.config = config
        self.server_as_prefix = server_as_prefix
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

        self.make_client_app_command()

    def make_client_app_command(self):
        command = f"python3  {self.dir_path}/client_application.py --client_id {self.id} " + \
                  f"--operation {self.operation} --ip_address {self.ip_address} " + \
                  f" --port_index {self.port_index} --port_status {self.port_status} "

        if self.operation.upper() == "GET" or self.operation.upper() == "TEST":
            command += f" --dest_path {self.destination_path} "
        if self.operation.upper() == "LIST":
            command += " --distributed "
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

        command += f" --server-as-prefix {self.server_as_prefix}"

        if self.env_gcc_required and self.env_gcc_source:
            command = "source {} && {} ".format(self.env_gcc_source, command)

        self.remote_execution_command = command

    def __del_dummy__(self):
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
        try:
            self.ssh_client_handler, stdin, stdout, stderr = remoteExecution(
                self.ip_address, self.username, self.password, self.remote_execution_command)
        except Exception as e:
            raise e

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

    def stop_process_on_remote_client(self):
        command = self.remote_execution_command + " --stop"
        try:
            remoteExecution(self.ip_address, self.username, self.password, command, blocking=True)
        except Exception as e:
            self.logger.excep(f"Exception in running the command {command} - {e}")
            raise e

    def stop(self, force_flag=False):
        """
        Send a message to the client node through ZMQ.
        Need to terminate forcefully.
        :return:
        """
        self.logger.info("Stopping client-{}".format(self.id))
        if force_flag:
            self.stop_process_on_remote_client()

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
    while not master.process_monitor_event.is_set():
        # Check for completion of indexing, Shutdown workers
        indexing_done = True
        # progress_bar("Object Count: {} ,Producer Msg Count: {}, Consumer Msg Count: {},
        # MSG-Queue Size - {}".format(master.index_data_count.value, master.index_msg_count.value,
        # master.received_index_msg_count.value, master.index_data_queue.qsize()) )
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
            if not monitors_stopped and master.monitor.monitor_index_distributor_status.value and \
                    master.monitor.monitor_status_poller_status.value and \
                    master.monitor.monitor_operation_progress_status.value:
                monitors_stopped = 1
                master.logger.info("All monitors belongs to Master terminated!")
            master.monitor.status_lock.release()

        if master.standalone:
            all_clients_completed = 1
            if not monitors_stopped and master.monitor.monitor_operation_progress_status.value:
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
        if not master.standalone and not workers_stopped and master.monitor.monitor_index_distributor_status.value:
            master.stop_workers()
            workers_stopped = 1

        # Once all the response received from Client Applications, shut down 3 monitors
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

    master.logger.info("Exiting process_put_operation with either PUTs completed or signal")


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
    distributed_listing = master.config.get("distributed", False)
    dump_object_keys_path = master.config.get("dest_path", None)
    while not master.process_monitor_event.is_set():
        progress_bar("Listed Object Keys - {}".format(master.index_data_count.value))

        if distributed_listing:
            # Check all the ClientApplications once they are finished
            all_clients_completed = 1
            for client in master.clients:
                # print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
                if not client.status.value:
                    if client.remote_client_status():
                        client.remote_client_exit_status()
                    else:
                        all_clients_completed = 0
            if all_clients_completed and master.monitor.monitor_status_poller_status.value == 1:
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info("Distributed LISTing is completed {} seconds".format(listing_time))
                if dump_object_keys_path:
                    master.listing_status.value = 2
                master.stop_workers()
                break
        else:
            # Standalone LIST: Check for completion of listing
            if master.listing_status.value == 1 and master.listing_progress.value == 0:
                listing_time = (datetime.now() - master.operation_start_time).seconds
                master.logger.info("LISTing is completed in {} seconds".format(listing_time))
                if dump_object_keys_path:
                    master.listing_status.value = 2
                else:
                    master.stop_workers()
                    break

            if master.listing_aggregation_status and master.listing_aggregation_status.value == 1:
                master.stop_workers()
                break
        time.sleep(1)

    master.logger.info("Total Object-Keys listed - {}".format(master.index_data_count.value))


def process_del_operation(master):
    workers_stopped = 0
    monitors_stopped = 0

    while not master.process_monitor_event.is_set():
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
        if not monitors_stopped and master.monitor.monitor_index_distributor_status.value and \
                master.monitor.monitor_status_poller_status.value and \
                master.monitor.monitor_operation_progress_status.value:
            monitors_stopped = 1
        master.monitor.status_lock.release()

        # Bring down workers.
        if not workers_stopped and master.monitor.monitor_index_distributor_status.value:
            master.stop_workers()
            workers_stopped = 1

        # Once all the response received from Client Applications, shut down 3 monitors
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

    while not master.process_monitor_event.is_set():
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
        if not monitors_stopped and master.monitor.monitor_index_distributor_status.value and \
                master.monitor.monitor_status_poller_status.value and \
                master.monitor.monitor_operation_progress_status.value:
            monitors_stopped = 1
        master.monitor.status_lock.release()

        # Bring down workers.
        if not workers_stopped and master.monitor.monitor_index_distributor_status.value:
            master.stop_workers()
            workers_stopped = 1

        # Once all the response received from Client Applications, shut down 3 monitors
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
    name = "DM_MasterApplication"
    prctl.set_name(name)
    prctl.set_proctitle(name)

    cli = CommandLineArgument()
    operation = cli.operation.upper()
    # Add signal handler
    signal_handler = SignalHandler()

    params = cli.options
    config_obj = Config(params)
    config = config_obj.get_config()

    master = Master(operation, config)
    if len(master.client_ip_list) == 0:
        print("No Clients provided. Exiting")
        sys.exit(-1)

    if config.get("debug", False):
        master.logging_level = "DEBUG"
    signal_handler.add_fn(master.set_proc_monitor_event)
    signal_handler.initiate()
    now = datetime.now()
    master.start()
    master.logger.info("DataMover Config Options : {}".format(config))
    master.logger.info("Started DataMover with Logger in {} mode!".format(master.logging_level))

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
    if not master.process_monitor_event.is_set() and config['compaction']:
        master.logger.info('Performing compaction')
        master.compaction()
    if master.process_monitor_event.is_set():
        master.logger.info('Stopping all processes as part of signal handler')
        master.stop()

    # Terminate logger at the end.
    master.stop_logging()  # Termination5
    print("INFO: Stopping master")
