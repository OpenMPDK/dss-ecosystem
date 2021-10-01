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
import os, sys
import time
from multiprocessing import Process, Queue, Value, Lock, Manager
from utils.utility import exception, exec_cmd, progress_bar, get_ip_address, is_queue_empty
from datetime import datetime
import json
import uuid
from logger import MultiprocessingLogger
from socket_communication import ClientSocket, ServerSocket
import prctl

"""
Monitor the progress of operation.
- Communicate to client , poll the status periodically.
- Process result 
- Display/Store result
"""
manager = Manager()
MONITOR_INACTIVE_WAIT_TIME = 1800 # 30 Mins
DEBUG_MESSAGE_INTERVAL = 120 # 10 Mins
PERSIST_FLUSH_INTERVAL = 60


class Monitor(object):

    def __init__(self, **kwargs):
        self.clients = kwargs.get("clients", [])
        self.config = kwargs["config"]
        self.prefix = self.config.get("prefix", None)  # Key prefix
        self.user_id = kwargs["config"]["client"]["user_id"]
        self.index_data_json_file = kwargs["index_data_json_file"]
        self.index_data_queue = kwargs["index_data_queue"]
        self.index_data_lock = kwargs["index_data_lock"]
        self.index_data_generation_complete = kwargs["index_data_generation_complete"]
        self.index_data_count = kwargs["index_data_count"]
        self.index_msg_count = kwargs["index_msg_count"]
        self.received_index_msg_count = kwargs["received_index_msg_count"]
        self.logger = kwargs["logger"]
        self.operation = kwargs["operation"]
        self.operation_start_time = kwargs["operation_start_time"]
        self.ip_address_family = self.config["ip_address_family"]
        self.standalone = kwargs["standalone"]

        if not self.standalone:
            self.status_queue = Queue()
        else:
            self.status_queue = kwargs['status_queue']
        self.status_lock = Lock()
        self.stop_status_poller = Value('i', 0)
        self.prefix_index_data = kwargs["prefix_index_data"]
        self.prefix_index_data_persist = dict(self.prefix_index_data.copy())

        self.all_index_data_distributed = Value('i', 0)

        # Listing
        self.listing_status = kwargs.get("listing_status", None)
        self.listing_aggregation_status = kwargs.get("listing_aggregation_status", None)
        self.listing_objectkey_queue = kwargs.get("listing_objectkey_queue", None)
        # Monitor termination
        self.monitor_index_data_sender = Value('i', 0)
        self.monitor_status_poller = Value('i', 0)
        self.monitor_progress_status = Value('i', 0)

        # Process
        self.process_index = None
        self.process_listing_aggregator = None

        # Unit Testcase
        self.testcase_passed = kwargs.get("testcase", False)

        # Index data queue tracking
        self.tracking_index_queue = Queue()

    def start(self):

        if self.operation.upper() == "LIST":
            self.process_listing_aggregator = Process(target=self.object_keys_aggregator)
            self.process_listing_aggregator.start()
        else:
            if not self.standalone:
                # Start index process
                self.process_index = Process(target=self.message_handler_index)
                self.process_index.start()

                # Start status_poller  process
                self.process_status = Process(target=self.message_handler_poller)
                self.process_status.start()
            else:
                self.logger.info('Running Monitor in standalone mode')

            # Start operation_progress process
            self.process_operation_progress = Process(target=self.operation_progress)
            self.process_operation_progress.start()

    def stop(self):

        self.logger.warn("Stopping monitors forcefully!")
        #self.status_lock.acquire()
        self.stop_status_poller.value = 1
        #self.status_lock.release()
        time.sleep(2)

        while self.process_listing_aggregator and self.process_listing_aggregator.is_alive():
            time.sleep(1)
            try:
                self.process_listing_aggregator.terminate()
            except Exception as e:
                self.logger.excep("Unable to terminate Listing Aggregator {}".format(e))

        if not self.standalone:
            while self.process_index and self.process_index.is_alive():
                time.sleep(1)
                try:
                    self.process_index.terminate()
                except Exception as e:
                    self.logger.excep("Unable to terminate MessageHandler - IndexSender {}".format(e))

            while self.process_status.is_alive():
                time.sleep(1)
                try:
                    self.process_status.terminate()
                except Exception as e:
                    self.logger.excep("Unable to terminate MessageHandler - StatusPoller ...\n {}".format(e))

        self.logger.warn("Terminated all running monitor processes ... ")

    def message_handler_index(self):
        """
        The "message_handler_index" function send index information to all the clients in round-robin fashion.
        message data = {"dir":"/bird/bird1", "files":[], "nfs_cluster":"10.1.51.54"}
        Stop Processing:
          Gracefully shut down monitor
          - All indexed is processed and thus "index_data_generation_complete" is set to 1
          - The "index_data_queue" has sent all outstanding messages to client application.
          Forcefully: If process is terminated
          - Forcefully stopped through "stop_status_poller" by setting it to 1.
        # TODO
            - In case socket-communication were not established with any clientApp, Monitor should stop
              and stop all other processes to shutdown DM.
        :return: None
        """
        name = "DM_monitor_message_handler_index"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        successful_socket_connection = 0
        for client in self.clients:
            try:
                client.socket_index = ClientSocket(self.logger, self.ip_address_family)
                client.socket_index.connect(client.ip_address, client.port_index)
                successful_socket_connection +=1
                self.logger.info("Connecting to INDEX MessageHandler {}:{}".format(client.ip_address, client.port_index))
            except ConnectionError as e:
                self.logger.excep("Socket Connection error for ClientApp-{} : {}".format(client.id, e))
                client.socket_index = None
        if successful_socket_connection == 0:
            self.logger.fatal("Monitor-Index: Socket connection was not established with any client. Exit!")
            # TODO Need to terminate master and along with all client-apps.
        # Buffer to store prefix index and file count  {"prefix":<file count>}
        first_index_distribution = 0
        index_distribution_start_time = datetime.now()
        message_count = 0
        object_count = 0
        debug_message_timer = datetime.now()
        client_count = len(self.clients)

        while True:
            # Forcefully stop the process
            if self.stop_status_poller.value :
                self.logger.error("Forcefully shutting down Monitor-index!")
                break

            previous_client_operation_status = 1
            data = {}

            if previous_client_operation_status:
                if self.index_data_queue.qsize() > 0:
                    data = self.index_data_queue.get()

            client_index  = self.received_index_msg_count.value % client_count
            client = self.clients[client_index]
                # Send data to ClientApplication running on a  Client-Physical Node
            if data:
                object_count_under_prefix = len(data["files"])
                if self.send_index_data(client, data):
                    previous_client_operation_status = 1
                    # Just for OPERATION stats collection
                    if first_index_distribution == 0:
                        index_distribution_start_time = datetime.now()
                        first_index_distribution = 1
                else:  # Re-send once again
                    previous_client_operation_status = 0
                    self.logger.error("Failed to send message to Client-{} ".format(client.id))

                    # Debug message , for success
                if previous_client_operation_status == 1:
                    message_count += 1
                    with self.received_index_msg_count.get_lock():
                        self.received_index_msg_count.value += 1

                    object_count += object_count_under_prefix
            # Debug message
            if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                self.logger.info("Messages distributed to clients-{}, Objects Count: {}".format(message_count,
                                                                                                object_count))
                debug_message_timer = datetime.now()

            if self.index_data_generation_complete.value == 1 and (self.index_msg_count.value == message_count):
                self.logger.info("Indexed data distribution is completed!")
                self.all_index_data_distributed.value = 1
                self.logger.info("Index Distribution FINISHED, time - {} Sec".format(
                    (datetime.now() - index_distribution_start_time).seconds))
                self.logger.info(
                    "Total distributed messages- {}, Objects Count-{}".format(message_count, object_count))

                # Inform all the client applications running on different nodes
                for client in self.clients:
                    data = {"indexing_done": 1}
                    if not self.send_index_data(client, data):
                        if not self.send_index_data(client, data):
                            self.logger.error(
                                "Unable to send indexing completion message to client-{}".format(client.id))
                    self.logger.info(
                        "Indexed data distribution is completed, Notifying ClientApplication {}:{} -> {}".format(
                            client.ip_address, client.port_index, data))
                # Intimidated all the clients, exit the loop.
                break

        # Close all sockets associated with clients.
        for client in self.clients:
            try:
                client.socket_index.close()
            except Exception as e:
                self.logger.excep("Monitor-index Closing Socket {}".format(e))

        self.monitor_index_data_sender.value = 1
        self.logger.info("Monitor-Index-Distribution is terminated gracefully! ")

    def persist_index_data(self):
        prefix_storage_file = self.index_data_json_file
        prefix_storage_file_new = prefix_storage_file + ".new"
        #prefix_index_data = json.dumps(self.prefix_index_data.copy())
        with self.index_data_count.get_lock():
            prefix_index_data = json.dumps(self.prefix_index_data_persist)

        self.logger.info(
            "Storing prefix_index_data to persistent storage - {}".format(prefix_storage_file))
        try:
            with open(prefix_storage_file_new, "w") as fh:
                fh.write(prefix_index_data)
        except Exception as e:
            self.logger.error("Exception in saving the prefix index data file with error {}".format(
                str(e)))
            raise
        else:
            try:
                if os.path.exists(prefix_storage_file):
                    os.replace(prefix_storage_file_new, prefix_storage_file)
                else:
                    os.rename(prefix_storage_file_new, prefix_storage_file)
            except Exception as e:
                self.logger.error("Exception in replacing the prefix index data file with error {}".format(
                    str(e)))
                raise


    def send_index_data(self, client, data):
        """
        Send index data for a socket client. On failure, resend once.
        The error code is returned by the client and captured here in the log.
        :param socket: client socket
        :param data: index data
        :return: success/failure 1/0
        """
        socket = client.socket_index
        ret = False
        try:
            ret = socket.send_json(data) # Send index data
        except RuntimeError as e:
            self.logger.excep("MSG send error - {}".format(e))
        except Exception as e:
            self.logger.excep("Monitor-Index -{}".format(e))

        return ret

    def message_handler_poller(self):
        """
        Monitor: Status Poller
        Collect the status from all the clients and add status to status_queue
        :return:
        """
        name = "DM_monitor_message_handler_poller"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        successful_socket_connection = 0
        for client in self.clients:
            try:
                client.socket_status = ClientSocket(self.logger, self.ip_address_family)
                client.socket_status.connect(client.ip_address, client.port_status)
                self.logger.info("Monitor-Status-Poller Connecting to client-app-{} tcp://{}:{}".format(client.id,
                                                                                                    client.ip_address,
                                                                                                    client.port_status))
                successful_socket_connection +=1
            except Exception as e:
                self.logger.excep("Monitor-Status-Poller: Socket connection error for ClientApp-{}".format(client.id, e))
                client.socket_status = None

        if successful_socket_connection == 0:
            self.logger.fatal("Monitor-Status-Poller: Socket connection was not established with any client. Exit!")
            # TODO Need to terminate master and along with all client-apps.
            # Need to eliminate client from client-list of connection was not successful.

        client_application_exit_count = 0
        received_status_msg_count = 0
        while True:
            # Condition to break the loop:
            # - Received all the operation status messages from all ClientApplications
            #           which matches with index/list object count
            # - Received a signal from master to shutdown

            if self.stop_status_poller.value == 1:
                break
            all_client_applications_terminated = True
            # Receive status message from all the ClientApplications.
            for client in self.clients:
                if client.socket_status is None:
                    continue
                # ERROR Handling: If the ClientApplication abruptly gets shutdown, exit code should be non-zero,
                #  We don't expect end message to be reached, hence the socket can be closed.
                if client.status.value and client.exit_status_code.value != 0:
                    client.socket_status.close()
                    client.socket_status = None
                    client_application_exit_count += 1
                    continue
                try:
                    all_client_applications_terminated = False
                    status = client.socket_status.recv_json()
                    if status:
                        # Check status message or end message for that Client
                        self.logger.debug(
                            "Monitor-Poller Operation Status for client-{}, Status - {}".format(client.id, status))
                        if "exit" in status and status["exit"]:
                            client.socket_status.close()
                            client.socket_status = None
                            client_application_exit_count += 1
                        else:
                            self.status_queue.put(status)
                            received_status_msg_count +=1
                except Exception as e:
                    self.logger.excep("Monitor-Status-Poller {} ".format(e))

            # Check if all client_applications terminated?
            if all_client_applications_terminated:
                self.logger.debug("Monitor-Status-Poller: All ClientApplications Terminated, Exit!")
                self.stop_status_poller.value = 1
                break
            # Exit monitor-status-poller if all messages received back from client-app.
            if self.all_index_data_distributed.value and ( self.index_msg_count.value == received_status_msg_count ):
                self.logger.info("Monitor-Status-Poller received all the status messages = {} , Exit!".format(received_status_msg_count))
                self.stop_status_poller.value = 1
                break

        # Closing all socket connection
        try:
            for client in self.clients:
                if client.socket_status:
                    client.socket_status.close()
        except Exception as e:
            self.logger.excep("Minitor-Poller {}".format(e))

        self.monitor_status_poller.value = 1
        self.logger.info("Monitor-Status-Poller terminated gracefully! ")

    @exception
    def operation_progress(self):
        """
        Monitor: Operation Progress
        Process the status result coming from different client nodes and update the progress.

        :return:
        """
        name = "DM_monitor_operation_progress"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        file_index_count = 0
        operation_success_count = 0  # S3 operation success count
        operation_failure_count = 0  # S3 operation failure count
        display_percentage = 10
        failure_file_size_in_byte = 0
        processed_prefix = {}  # A hash which store the prefixes those have been exercised from S3
        debug_message_timer = datetime.now()
        last_flush_timer = time.time()
        original_file_size_in_byte = 0
        prefix_dir_deleted = []  # Contains all prefix dir erased from S3 during DEL operation.

        while True:

            if self.standalone:
                if self.index_data_generation_complete.value == 1:
                    if self.all_index_data_distributed.value != 1:
                        self.logger.debug("Index generation completed")
                    self.all_index_data_distributed.value = 1

            # Stop the monitor when status-poller is stopped and status_queue is empty.
            if self.stop_status_poller.value == 1 and is_queue_empty(self.status_queue):
                self.logger.info("Monitor-Operation-Progress is about to exit!")
                break

            if not is_queue_empty(self.status_queue):
                status = self.status_queue.get()
                operation_success_count += status.get("success", 0)
                if status.get("failure", 0):
                    operation_failure_count += status.get("failure", 0)
                    failure_file_size_in_byte += status.get("size", 0)

                # Progress calculation, the value of index_data_count increases as indexing at progresses.
                # The file_index_count increases as response received from client application.
                prefix = status["dir"]
                if self.operation.upper() == "PUT":
                    if status["dir"].startswith("/"):
                        prefix = status["dir"][1:]
                    if not status["dir"].endswith("/"):
                        prefix += "/"
                if prefix in processed_prefix:
                    processed_prefix[prefix]["success"] += status.get("success", 0)
                    processed_prefix[prefix]["failure"] += status.get("failure", 0)
                else:
                    processed_prefix[prefix] = {"success" : status.get("success", 0) , "failure" : status.get("failure", 0) }

                if self.operation.upper() == "PUT" and prefix in processed_prefix:
                    processed_files_count = processed_prefix[prefix]["success"] + processed_prefix[prefix]["failure"]
                    if self.prefix_index_data[prefix]["files"] == processed_files_count:
                        prefix_data = self.prefix_index_data[prefix]
                        prefix_data["succeeded"] = processed_prefix[prefix]["success"]

                        with self.index_data_count.get_lock():
                            self.prefix_index_data_persist[prefix] = prefix_data
                        self.logger.debug('Index data - Dir {} with file count {} fully uploaded'.format(
                            prefix, self.prefix_index_data[prefix]["files"]))
                        if int(time.time() - last_flush_timer) > PERSIST_FLUSH_INTERVAL:
                            try:
                                self.persist_index_data()
                                self.logger.info('Persisting the index data')
                                last_flush_timer = time.time()
                            except Exception as e:
                                self.logger.fatal("Exception in persisting index data - {}".format(str(e)))
                                # TODO: BAIL OUT
                elif  self.operation.upper() in ["DEL"]:
                    if self.prefix_index_data[prefix]["files"] == processed_prefix[prefix]["success"]:
                        prefix_dir_deleted.append(prefix)

                file_index_count = operation_success_count + operation_failure_count
                # Debug message
                if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                    self.logger.info(
                        "OperationProgress: Indexed Files={}, Operation (Success={}, Failure={})".format(
                            self.index_data_count.value, operation_success_count,
                            operation_failure_count))
                    debug_message_timer = datetime.now()
                if self.all_index_data_distributed.value and self.index_data_count.value:

                    upload_percentage = (file_index_count / self.index_data_count.value) * 100
                    if upload_percentage > display_percentage:
                        self.logger.info(
                            " ** Monitor-Progress - Operation Progress Status - {:.2f}% **".format(upload_percentage))
                        display_percentage += 10

            # All index data distributed to clients and received all operation status back from clients.
            if self.all_index_data_distributed.value and  file_index_count ==  self.index_data_count.value:
                self.stop_status_poller.value = 1   # This will stop Monitor-Poller
                self.logger.info("Monitor-Operation-Progress received status of all objects = {}".format(file_index_count))
        
        if self.operation.upper() == "PUT" and self.prefix_index_data:
            try:
                self.persist_index_data()
                self.logger.info('Persisting the final index data')
            except Exception as e:
                self.logger.fatal("Exception in persisting final index data - {}".format(str(e)))

        total_operation_time = (datetime.now() - self.operation_start_time).seconds

        # Calculate operation BandWidth
        # Calculate total operation(PUT/GET/DEL) size
        if self.prefix:
            for prefix, value in self.prefix_index_data_persist.items():
                if self.operation.upper() == "PUT":
                    fields = prefix.split("/")[1:]
                    prefix = "/".join(fields)

                if prefix.startswith(self.prefix):
                    original_file_size_in_byte += value["size"]
        else:
            for prefix, value in self.prefix_index_data_persist.items():
                if prefix in processed_prefix:
                    original_file_size_in_byte += value["size"]

        #  Update or Remove metadata-persistent-index file.
        if self.operation.upper() == "DEL" and self.prefix_index_data_persist:
            for prefix in prefix_dir_deleted:
                if prefix in self.prefix_index_data_persist:
                    del self.prefix_index_data_persist[prefix]

            if self.prefix_index_data_persist:
                try:
                    self.persist_index_data()
                except Exception as e:
                    self.logger.fatal("Exception in persisting final index data - {}".format(str(e)))
            else:
                # Remove persistent index file.
                os.remove(self.index_data_json_file)
                self.logger.warn("All the objects removed from S3. Removed persistent index file - {}".format(
                    self.index_data_json_file))

        success_operation_size_in_byte = original_file_size_in_byte

        self.logger.info("***** Operation Statistics *****")
        if operation_success_count and self.index_data_count.value:
            success_percentage = (operation_success_count / self.index_data_count.value) * 100
            self.logger.info("Total {} Operation: {},  Operation Success:{} - {:.2f}%".format(self.operation,
                                                                                              self.index_data_count.value,
                                                                                              operation_success_count,
                                                                                              success_percentage))
        if operation_failure_count and self.index_data_count.value:
            failure_percentage = (operation_failure_count / self.index_data_count.value) * 100
            self.logger.info("Total {} Operation:{}, Operation Failure:{} - {:.2f}%".format(self.operation,
                                                                                            self.index_data_count.value,
                                                                                            operation_failure_count,
                                                                                            failure_percentage))
            success_operation_size_in_byte -= failure_file_size_in_byte
            self.logger.warn("DataMover RESUME operation is required!")

        bandwidth = 0
        if total_operation_time:
            bandwidth = success_operation_size_in_byte / (1024 * 1024 * total_operation_time)
        operation_size_in_gb = success_operation_size_in_byte / (1024 * 1024 * 1024)

        self.logger.info(
            "Operation {} completed in {} seconds for {:.2f} GiB ".format(self.operation, total_operation_time,
                                                                          operation_size_in_gb))
        self.logger.info("Operation {} BandWidth = {:.2f} MiB/sec ".format(self.operation, bandwidth))
        self.monitor_progress_status.value = 1
        # Check if TestCase has passed
        if self.index_data_count.value == operation_success_count:
            self.testcase_passed.value = True
        self.logger.info("Monitor-Operation-Progress terminated! ")

    def object_keys_aggregator(self):
        name = "DM_monitor_obj_keys_aggregator"
        prctl.set_name(name)
        prctl.set_proctitle(name)

        listing_path = self.logger.path
        if self.config["dest_path"]:
            listing_path = self.config["dest_path"]

        listing_file = listing_path + "/listing_object_keys"
        fh = None
        try:
            if not os.path.isdir(listing_path):
                command = "mkdir -p {}".format(listing_path)
                ret, console = exec_cmd(command, True, True, self.user_id)
                if ret:
                    self.logger.error(
                        "Failed to create listing path {} \n ret-{}, {}".format(listing_path, ret, console))
                    self.listing_aggregation_status.value = 1
                    sys.exit()

            if os.path.exists(listing_file):
                os.remove(listing_file)
            fh = open(listing_file, "w")
            if not fh:
                self.logger.error("Failed to open file \"{}\" to dump listing object keys".format(listing_file))
                return
            self.logger.debug("ObjectKeys aggregation started!")
            while True:
                if self.listing_status and self.listing_status.value == 2 and is_queue_empty(self.listing_objectkey_queue):
                    self.listing_aggregation_status.value = 1
                    self.logger.debug("Object Keys aggregation is completed!")
                    break
                if not is_queue_empty(self.listing_objectkey_queue):
                    object_keys_data = self.listing_objectkey_queue.get()
                    prefix = object_keys_data["prefix"]
                    object_keys = object_keys_data["object_keys"]
                    for object_key in object_keys:
                        fh.write(prefix + object_key + "\n")

            self.logger.info("Listing ObjectKeys are dumped at - {}".format(listing_file))
        except Exception as e:
            self.logger.excep("ListingAggregation: {}".format(e))
        finally:
            if fh:
                fh.close()


