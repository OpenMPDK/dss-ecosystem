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
import time
import  zmq
from multiprocessing import Process,Queue,Value, Lock, Manager
from utils.utility import exception, exec_cmd, progress_bar, get_ip_address
from datetime import datetime
import json
from logger import  MultiprocessingLogger
from socket_communication import ClientSocket

"""
Monitor the progress of operation.
- Communicate to client , poll the status periodically.
- Process result 
- Display/Store result
"""
manager = Manager()
MONITOR_INACTIVE_WAIT_TIME = 180 # 30 Mins
DEBUG_MESSAGE_INTERVAL = 100 # 10 Mins

class Monitor:

    def  __init__(self, **kwargs):
        self.clients = kwargs.get("clients",[])
        self.config = kwargs["config"]
        self.prefix = self.config.get("prefix", None)  # Key prefix
        self.user_id = kwargs["config"]["client"]["user_id"]
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

        self.status_queue = Queue()
        self.status_lock  = Lock()
        self.stop_status_poller = Value('i', 0)
        self.prefix_index_data = manager.dict()

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

    def start(self):

        if self.operation.upper() == "LIST":
          self.process_listing_aggregator = Process(target=self.object_keys_aggregator)
          self.process_listing_aggregator.start()
        else:
          # Start index process
          self.process_index = Process(target=self.message_handler_index1)
          self.process_index.start()

          # Start status_poller  process
          self.process_status = Process(target=self.message_handler_poller)
          self.process_status.start()

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

        :return: None
        """
        try:
            context = zmq.Context()
            for client in self.clients:
                client.socket_index = context.socket(zmq.REQ)
                if self.ip_address_family.upper() == "IPV6":
                    client.socket_index.setsockopt(zmq.IPV6, 1)
                client.socket_index.connect("tcp://{}:{}".format(client.ip_address, client.port_index))
                self.logger.info("Connecting to INDEX MessageHandler tcp://{}:{}".format(client.ip_address, client.port_index))
        except Exception as e:
            self.logger.excep("ZMQ Connection error IndexDistributor - {}".format(e))
            return

        # Buffer to store prefix index and file count  {"prefix":<file count>}
        first_index_distribution  = 0
        index_distribution_start_time = datetime.now()
        message_count = 0
        object_count = 0


        while True:

            # Forcefully stop the process
            #self.status_lock.acquire()
            stop = self.stop_status_poller.value
            #self.status_lock.release()
            if stop :
                self.logger.error("Forcefully shutting down Monitor-index!")
                break

            previous_client_operation_status = 1
            data = {}
            # Send index data to each client in round-robin fashion
            for client in self.clients:
                # Get data from shared index queue, if the previous client processed data successfully
                if previous_client_operation_status:
                    data = {}
                    if self.index_data_queue.qsize() > 0:
                        data = self.index_data_queue.get()
                # Send data to ClientApplication running on a  Client-Physical Node
                if data:
                    object_count_under_prefix = len(data["files"])
                    # Buffer prefix_index_data for persistent storage only to be used during PUT
                    if self.operation.upper() == "PUT":
                        object_prefix_key = data["dir"][1:] + "/"
                        if object_prefix_key in self.prefix_index_data:
                            self.prefix_index_data[object_prefix_key]["files"] += object_count_under_prefix
                            self.prefix_index_data[object_prefix_key]["size"]  += data["size"]
                        else:
                            self.prefix_index_data[object_prefix_key] = manager.dict()
                            self.prefix_index_data[object_prefix_key].update({"files": len(data["files"]), "size": data["size"]})

                    # self.logger.debug("Sending index data - {}:{} -> {}".format(client.ip_address, client.port_index, data))
                    if self.send_index_data(client, data):
                        previous_client_operation_status = 1
                        # Just for OPERATION stats collection
                        if first_index_distribution == 0:
                            index_distribution_start_time = datetime.now()
                            first_index_distribution =1
                    else: # Re-send once again
                        previous_client_operation_status = 0
                        if client.socket_index.closed:
                            client.socket_index = context.socket(zmq.REQ)
                            client.socket_index.connect("tcp://{}:{}".format(client.ip_address, client.port_index))
                            self.logger.info("Refreshed the socket-index for the Client-{} : {}".format(client.id, client.ip_address))
                        if self.send_index_data(client, data):
                            previous_client_operation_status = 1
                        else:
                            self.logger.error("Failed to send message to Client-{} ".format(client.id))

                    # Debug message , for success
                    if previous_client_operation_status == 1:
                        message_count += 1
                        with self.received_index_msg_count.get_lock():
                          self.received_index_msg_count.value +=1

                        object_count += object_count_under_prefix

            # Debug message
            if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                self.logger.info("Messages distributed to clients-{}, Objects Count: {}".format(message_count,
                                                                                             object_count))
                debug_message_timer = datetime.now()
            if self.index_data_generation_complete.value == 1  and (self.index_msg_count.value == message_count) :
                self.logger.info("Indexed data distribution is completed!")
                self.all_index_data_distributed.value = 1
                self.logger.info("Index Distribution FINISHED, time - {} Sec".format((datetime.now() - index_distribution_start_time).seconds))
                self.logger.info("Total distributed messages- {}, Objects Count-{}".format(message_count, object_count))

                # Inform all the client applications running on different nodes
                for client in self.clients:
                    data = {"indexing_done": 1}
                    if not self.send_index_data(client, data):
                        if not self.send_index_data(client, data):
                            self.logger.error("Unable to send indexing completion message to client-{}".format(client.id))
                    self.logger.info("Indexed data distribution is completed, Notifying ClientApplication {}:{} -> {}".format(client.ip_address, client.port_index, data))
                # Intimidated all the clients, exit the loop.
                break

        # Closing all socket connection
        try:
            # Close all sockets associated with clients.
            for client in self.clients:
                client.socket_index.close()
        except Exception as e:
            self.logger.excep("{}: monitor-index Closing Socket {}".format(e))
        finally:
            context.term() # Terminate context


        # Update MasterApplication about termination of Monitor
        #self.monitor_index_data_sender.value = 1

        # Storing prefix index data to persistent storage
        if self.operation.upper() == "PUT":
            prefix_storage_file = "/var/log/prefix_index_data.json"
            prefix_index_data = {}
            for key,value in self.prefix_index_data.items():
                prefix_index_data[key] = value.copy()

            self.logger.info("Storing prefix_index_data to persistent storage - {}".format(prefix_storage_file))
            try:
                with open(prefix_storage_file, "w") as persistent_storage:
                    json.dump(prefix_index_data, persistent_storage)
                self.logger.info("INFO: Stored file index data to {}".format(prefix_storage_file))
            except Exception as e:
                self.logger.info("Dump Persistent Data - {}".format(e))

        self.monitor_index_data_sender.value = 1
        self.logger.info("Monitor-Index-Distribution is terminated gracefully! ")


    def message_handler_index1(self):
        """
        The "message_handler_index" function send index information to all the clients in round-robin fashion.
        message data = {"dir":"/bird/bird1", "files":[], "nfs_cluster":"10.1.51.54"}
        Stop Processing:
          Gracefully shut down monitor
          - All indexed is processed and thus "index_data_generation_complete" is set to 1
          - The "index_data_queue" has sent all outstanding messages to client application.
          Forcefully: If process is terminated
          - Forcefully stopped through "stop_status_poller" by setting it to 1.

        :return: None
        """
        try:
            for client in self.clients:
                client.socket_index = ClientSocket(self.logger)
                client.socket_index.connect(client.ip_address, client.port_index)
                self.logger.info("Connecting to INDEX MessageHandler {}:{}".format(client.ip_address, client.port_index))
        except ConnectionError as e:
            self.logger.excep("Socket Connection error: {}".format(e))
            return

        # Buffer to store prefix index and file count  {"prefix":<file count>}
        first_index_distribution  = 0
        index_distribution_start_time = datetime.now()
        message_count = 0
        object_count = 0
        debug_message_timer = datetime.now()

        while True:

            # Forcefully stop the process
            if self.stop_status_poller.value :
                self.logger.error("Forcefully shutting down Monitor-index!")
                break

            previous_client_operation_status = 1
            data = {}
            # Send index data to each client in round-robin fashion
            for client in self.clients:
                # Get data from shared index queue, if the previous client processed data successfully
                if previous_client_operation_status:
                    data = {}
                    if self.index_data_queue.qsize() > 0:
                        data = self.index_data_queue.get()
                # Send data to ClientApplication running on a  Client-Physical Node
                if data:
                    object_count_under_prefix = len(data["files"])
                    # Buffer prefix_index_data for persistent storage only to be used during PUT
                    if self.operation.upper() == "PUT":
                        object_prefix_key = data["dir"][1:] + "/"
                        if object_prefix_key in self.prefix_index_data:
                            self.prefix_index_data[object_prefix_key]["files"] += object_count_under_prefix
                            self.prefix_index_data[object_prefix_key]["size"]  += data["size"]
                        else:
                            self.prefix_index_data[object_prefix_key] = manager.dict()
                            self.prefix_index_data[object_prefix_key].update({"files": len(data["files"]), "size": data["size"]})

                    # self.logger.debug("Sending index data - {}:{} -> {}".format(client.ip_address, client.port_index, data))
                    if self.send_index_data1(client, data):
                        previous_client_operation_status = 1
                        # Just for OPERATION stats collection
                        if first_index_distribution == 0:
                            index_distribution_start_time = datetime.now()
                            first_index_distribution =1
                    else: # Re-send once again
                        previous_client_operation_status = 0
                        self.logger.error("Failed to send message to Client-{} ".format(client.id))

                    # Debug message , for success
                    if previous_client_operation_status == 1:
                        message_count += 1
                        with self.received_index_msg_count.get_lock():
                          self.received_index_msg_count.value +=1

                        object_count += object_count_under_prefix

            # Debug message
            if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                self.logger.info("Messages distributed to clients-{}, Objects Count: {}".format(message_count,
                                                                                             object_count))
                debug_message_timer = datetime.now()
            if self.index_data_generation_complete.value == 1  and (self.index_msg_count.value == message_count) :
                self.logger.info("Indexed data distribution is completed!")
                self.all_index_data_distributed.value = 1
                self.logger.info("Index Distribution FINISHED, time - {} Sec".format((datetime.now() - index_distribution_start_time).seconds))
                self.logger.info("Total distributed messages- {}, Objects Count-{}".format(message_count, object_count))

                # Inform all the client applications running on different nodes
                for client in self.clients:
                    data = {"indexing_done": 1}
                    if not self.send_index_data1(client, data):
                        if not self.send_index_data1(client, data):
                            self.logger.error("Unable to send indexing completion message to client-{}".format(client.id))
                    self.logger.info("Indexed data distribution is completed, Notifying ClientApplication {}:{} -> {}".format(client.ip_address, client.port_index, data))
                # Intimidated all the clients, exit the loop.
                break

        # Closing all socket connection
        try:
            # Close all sockets associated with clients.
            for client in self.clients:
                client.socket_index.close()
        except Exception as e:
            self.logger.excep("{}: monitor-index Closing Socket {}".format(e))



        # Update MasterApplication about termination of Monitor
        #self.monitor_index_data_sender.value = 1

        # Storing prefix index data to persistent storage
        if self.operation.upper() == "PUT":
            prefix_storage_file = "/var/log/prefix_index_data.json"
            prefix_index_data = {}
            for key,value in self.prefix_index_data.items():
                prefix_index_data[key] = value.copy()

            self.logger.info("Storing prefix_index_data to persistent storage - {}".format(prefix_storage_file))
            try:
                with open(prefix_storage_file, "w") as persistent_storage:
                    json.dump(prefix_index_data, persistent_storage)
                self.logger.info("INFO: Stored file index data to {}".format(prefix_storage_file))
            except Exception as e:
                self.logger.info("Dump Persistent Data - {}".format(e))

        self.monitor_index_data_sender.value = 1
        self.logger.info("Monitor-Index-Distribution is terminated gracefully! ")


    def send_index_data1(self, client, data):
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


    def send_index_data(self, client, data):
        """
        Send index data for a socket client. On failure, resend once.
        The error code is returned by the client and captured here in the log.
        :param socket: client socket
        :param data: index data
        :return: success/failure 1/0
        """
        status = {}
        socket = client.socket_index
        try:
            socket.send_json(data) # Send index data
            # Wait maximum (30sec) for ClientApplication's response. Otherwise re-send index-data.
            index = 0
            while index < 30 : # Retry 30 times at max
              received_response = socket.poll(timeout=1000)  # Wait 1 sec
              if received_response:
                status = socket.recv_json()
                break
              index +=1
        except Exception as e:
            #self.logger.excep("Monitor-Index -{}".format(e))
            socket.close()
            self.logger.info("Closed socket for Client-{}:{}".format(client.id,client.ip_address))

        # status = {"success": 1} or {"success": 0}  for failure, try second time
        if status.get("success", False):
          if status["success"] == 1:
            return 1
          elif status["success"] == 0:
            self.logger.error("ERROR:{}".format(status["ERROR"]))

        return 0


    def message_handler_poller(self):
        """
        Monitor: Status Poller
        Collect the status from all the clients and add status to status_queue
        :return:
        """
        try:
            context = zmq.Context()
            for client in self.clients:
                client.socket_status = context.socket(zmq.PULL)
                if self.ip_address_family.upper() == "IPV6":
                    client.socket_status.setsockopt(zmq.IPV6, 1)
                self.logger.info("Monitor-Poller Connecting to client-app-{} tcp://{}:{}".format(client.id, client.ip_address, client.port_status))
                client.socket_status.connect("tcp://{}:{}".format(client.ip_address, client.port_status))
        except Exception as e:
            self.logger.excep("ZMQ Connection ERROR, StatusPoller - {}".format(e))
            return

        client_application_exit_count = 0
        while True:

            ## Condition to break the loop:
            # - Received all the operation status messages from all ClientApplications which matches with index/list object count
            # - Received a signal from master to shutdown

            if self.stop_status_poller.value == 1:
                break
            all_client_applications_terminated = True
            # Receive status message from all the ClientApplications.
            for client in self.clients:
                if client.socket_status is None:
                    continue
                ## ERROR Handling: If the ClientApplication abruptly gets shutdown, exit code should be non-zero,
                #  We don't except for end message to be reached, hence the socket can be closed.
                if client.status.value and client.exit_status_code.value != 0:
                    client.socket_status.close()
                    client.socket_status = None
                    client_application_exit_count += 1
                    continue
                try:
                    all_client_applications_terminated = False
                    received_response = client.socket_status.poll(timeout=1000)  # Wait 1 secs
                    if received_response:
                        status = client.socket_status.recv_json()
                        # Check status message or end message for that Client
                        self.logger.debug("Monitor-Poller Operation Status for client-{}, Status - {}".format(client.id, status))
                        if "exit" in status and status["exit"]:
                            client.socket_status.close()
                            client.socket_status = None
                            client_application_exit_count += 1
                        else:
                            self.status_queue.put(status)
                except Exception as e:
                    self.logger.excep("Monitor-Status-Poller {} ".format(e))


            # Check if all client_applications terminated?
            if all_client_applications_terminated:
                self.logger.debug("Monitor-Status-Poller: All ClientApplications Terminated, EXITs! ")
                self.stop_status_poller.value = 1
                break

        # Closing all socket connection
        try:
            for client in self.clients:
                if client.socket_status:
                    client.socket_status.close()
        except Exception as e:
            self.logger.excep("Minitor-Poller {}".format(e))
        finally:
            context.term() # Terminate context.

        #self.status_lock.acquire()
        self.monitor_status_poller.value = 1
        #self.status_lock.release()
        self.logger.info("Monitor-Status-Poller terminated gracefully! ")


    @exception
    def operation_progress(self):
        """
        Monitor: Operation Progress
        Process the status result coming from different client nodes and update the progress.

        :return:
        """
        file_index_count = 0
        operation_success_count = 0 # S3 operation success count
        operation_failure_count = 0 # S3 operation failure count
        display_percentage = 10
        failure_file_size_in_byte = 0
        processed_prefix = {} # A hash which store the prefixes those have been exercised from S3
        debug_message_timer = datetime.now()

        while True:
            # Condition to break the loop:
            # If status poller has stopped  and status_queue is empty

            # Forcefully stop the process
            if self.stop_status_poller.value == 1 and self.status_queue.qsize() == 0:
                break

            if self.status_queue.qsize() > 0:
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
                if status["dir"] in processed_prefix:
                    processed_prefix[prefix] +=  status.get("success", 0) + status.get("failure", 0)
                else:
                    processed_prefix[prefix] = status.get("success", 0) + status.get("failure", 0)

                file_index_count = operation_success_count + operation_failure_count
                #self.logger.debug("OperationProgress: Total Files-{}, Success-{}, Failure-{}".format(self.index_data_count.value,
                #                                                                  operation_success_count,
                #                                                                  operation_failure_count))
                # Debug message
                if (datetime.now() - debug_message_timer).seconds > DEBUG_MESSAGE_INTERVAL:
                    self.logger.info("OperationProgress: Indexed Files={}, Operation (Success={}, Failure={})".format(
                                                                                           self.index_data_count.value,
                                                                                           operation_success_count,
                                                                                           operation_failure_count))
                    debug_message_timer = datetime.now()
                if self.all_index_data_distributed.value and self.index_data_count.value:

                    upload_percentage = (file_index_count / self.index_data_count.value) * 100
                    if upload_percentage > display_percentage:
                        self.logger.info(" ** Monitor-Progress - Operation Progress Statuss - {:.2f}% **".format(upload_percentage))
                        display_percentage +=10

            ## All index data distributed to clients and received all operation status back from clients.
            if self.all_index_data_distributed.value and  file_index_count ==  self.index_data_count.value:
                self.stop_status_poller.value = 1   # This will stop Monitor-Poller

        total_operation_time = (datetime.now() - self.operation_start_time).seconds

        # Calculate operation BandWidth
        if self.operation.upper() == "DEL" or self.operation.upper() == "GET":
            prefix_index_data_file = "/var/log/prefix_index_data.json"
            with open(prefix_index_data_file, "r") as prefix_index_data_handler:
                prefix_index_data = json.load(prefix_index_data_handler)
        else:
            prefix_index_data = self.prefix_index_data

        # Calculate total operation(PUT/GET/DEL) size
        original_file_size_in_byte = 0
        if self.prefix:
          for prefix, value in prefix_index_data.items():
            if self.operation.upper() == "PUT":
              fields = prefix.split("/")[1:]
              prefix = "/".join(fields) 
             
            if prefix.startswith(self.prefix):
              original_file_size_in_byte += value["size"]
        else:
          for prefix,value in prefix_index_data.items():
            if prefix in processed_prefix:
              original_file_size_in_byte += value["size"]

        success_operation_size_in_byte= original_file_size_in_byte

        self.logger.info("***** Operation Statistics *****")
        if operation_success_count and self.index_data_count.value:
            success_percentage = (operation_success_count / self.index_data_count.value) * 100
            self.logger.info("Total {} Operation: {},  Operation Success:{} - {:.2f}%".format(self.operation, self.index_data_count.value, operation_success_count, success_percentage))
        if operation_failure_count and self.index_data_count.value:
            failure_percentage = (operation_failure_count / self.index_data_count.value) * 100
            self.logger.info("Total {} Operation:{}, Operation Failure:{} - {:.2f}%".format(self.operation, self.index_data_count.value, operation_failure_count,failure_percentage))
            success_operation_size_in_byte -= failure_file_size_in_byte
            self.logger.warn("DataMover RESUME operation is required!")

        bandwidth = 0
        if total_operation_time:
            bandwidth = success_operation_size_in_byte / ( 1024 * 1024 * total_operation_time)
        operation_size_in_gb = success_operation_size_in_byte / ( 1024 * 1024 * 1024)

        self.logger.info("Operation {} completed in {} seconds for {:.2f} GiB ".format(self.operation,  total_operation_time, operation_size_in_gb))
        self.logger.info("Operation {} BandWidth = {:.2f} MiB/sec ".format(self.operation, bandwidth))
        self.monitor_progress_status.value = 1
        # Check if TestCase has passed
        if self.index_data_count.value == operation_success_count:
            self.testcase_passed.value = True
        self.logger.info("Monitor-Progress-Status terminated! ")


    def object_keys_aggregator(self):

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
              self.logger.error("Failed to create listing path {} \n ret-{}, {}".format(listing_path, ret, console))
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
            if self.listing_status and self.listing_status.value == 2 and self.listing_objectkey_queue.qsize() == 0:
              self.listing_aggregation_status.value = 1
              self.logger.debug("Object Keys aggregation is completed!")
              break
            if self.listing_objectkey_queue.qsize() > 0:
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


