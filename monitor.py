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
from datetime import datetime
import json
from logger import  MultiprocessingLogger

"""
Monitor the progress of operation.
- Communicate to client , poll the status periodically.
- Process result 
- Display/Store result
"""
manager = Manager()

class Monitor:

    def  __init__(self, **kwargs):
        self.clients = kwargs.get("clients",[])
        self.index_data_queue = kwargs["index_data_queue"]
        self.index_data_lock = kwargs["index_data_lock"]
        self.index_data_generation_complete = kwargs["index_data_generation_complete"]
        self.index_data_count = kwargs["index_data_count"]
        self.logger = kwargs["logger"]
        self.operation = kwargs["operation"]
        self.operation_start_time = kwargs["operation_start_time"]

        self.status_queue = Queue()
        self.status_lock  = Lock()
        self.stop_status_poller = Value('i', 0)
        self.prefix_index_data = manager.dict()

        self.all_index_data_distributed = Value('i', 0)

        # Monitor termination
        self.monitor_index_data_sender = Value('i', 0)
        self.monitor_status_poller = Value('i', 0)
        self.monitor_progress_status = Value('i', 0)


    def start(self):

        # Start index process
        self.process_index = Process(target=self.message_handler_index)
        self.process_index.start()

        # Start status_poller  process
        self.process_status = Process(target=self.message_handler_poller)
        self.process_status.start()

        # Start operation_progress process
        self.process_operation_progress = Process(target=self.operation_progress)
        self.process_operation_progress.start()



    def stop(self):

        self.logger.debug("Stopping MessageHandler - ")
        self.status_lock.acquire()
        self.stop_status_poller.value = 1
        self.status_lock.release()
        time.sleep(2)

        while self.process_index.is_alive():
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

        self.logger.info("Stopped all monitor processes ... ")
        #print("INFO: All monitor stopped ...")


    def display(self):
        pass

    def result(self):
        pass

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

        context = zmq.Context()

        for client in self.clients:
            client.socket_index = context.socket(zmq.REQ)
            #print("INFO: Connecting to INDEX MessageHandler tcp://{}:{}".format(client.ip, client.port_index))
            self.logger.info("Connecting to INDEX MessageHandler tcp://{}:{}".format(client.ip, client.port_index))
            client.socket_index.connect("tcp://{}:{}".format(client.ip, client.port_index))

        # Buffer to store prefix index and file count  {"prefix":<file count>}
        first_index_distribution  = 0
        index_distribution_start_time = datetime.now()

        while True:

            # Forcefully stop the process
            self.status_lock.acquire()
            stop = self.stop_status_poller.value
            self.status_lock.release()
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
                    if not self.index_data_queue.empty():
                        data = self.index_data_queue.get()
                # Send data to ClientApplication running on a  Client-Physical Node
                if data:
                    # Buffer prefix_index_data for persistent storage only to be used during PUT
                    if self.operation.upper() == "PUT":
                        object_prefix_key = data["dir"][1:] + "/"
                        if object_prefix_key in self.prefix_index_data:
                            self.prefix_index_data[object_prefix_key]["files"] += len(data["files"])
                            self.prefix_index_data[object_prefix_key]["size"]  += data["size"]
                        else:
                            self.prefix_index_data[object_prefix_key] = manager.dict()
                            self.prefix_index_data[object_prefix_key].update({"files": len(data["files"]), "size": data["size"]})

                    #self.logger.debug("Sending index data - {}:{} -> {}".format(client.ip, client.port_index, data))
                    if self.send_index_data(client, data):
                        #self.index_data_count.value += len(data.get("files", []))
                        previous_client_operation_status = 1
                        # Just for OPERATION stats collection
                        if first_index_distribution == 0:
                            index_distribution_start_time = datetime.now()
                            first_index_distribution =1
                    else: # Re-send once again
                        previous_client_operation_status = 0
                        if client.socket_index.closed:
                            client.socket_index = context.socket(zmq.REQ)
                            client.socket_index.connect("tcp://{}:{}".format(client.ip, client.port_index))
                            self.logger.info("Refreshed the socket-index for the Client-{} : {}".format(client.id, client.ip))
                        if self.send_index_data(client, data):
                            #self.index_data_count.value += len(data.get("files", []))
                            previous_client_operation_status = 1

            if self.index_data_generation_complete.value == 1  and self.index_data_queue.empty():
                self.logger.info("Indexed data distribution is completed!")
                self.all_index_data_distributed.value = 1
                self.logger.info("Index Distribution FINISHED,  time - {}".format((datetime.now() - index_distribution_start_time).seconds))

                # Inform all the client applications running on different nodes
                for client in self.clients:
                    data = {"indexing_done": 1}
                    if not self.send_index_data(client, data):
                        if not self.send_index_data(client, data):
                            self.logger.error("Unable to send indexing completion message to client-{}".format(client.id))
                    self.logger.info("Indexed data distribution is completed, Notifying ClientApplication {}:{} -> {}".format(client.ip, client.port_index, data))
                # Intimidated all the clients, exit the loop.
                break

        # Closing all socket connection
        try:
            # Close all sockets associated with clients.
            for client in self.clients:
                client.socket_index.close()
            # Terminate context
            context.term()

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



    def send_index_data(self, client, data):
        """
        Send index data for a socket client. On failure, resend once.
        :param socket: client socket
        :param data: index data
        :return: success/failure 1/0
        """
        status = {}
        socket = client.socket_index
        try:
            socket.send_json(data) # Send index data
            # Wait (3sec) for ClientApplication's response on operation. Otherwise re-send index data.
            received_response = socket.poll(timeout=1500)  # Wait 3 secs
            if received_response:
                status = socket.recv_json()
            else:
                self.logger.warn("Monitor-Index -- RSP not received from ClientApp - {}".format(status))
        except Exception as e:
            self.logger.excep("Monitor-Index -{}".format(e))
            #print("EXCEPTION: Monitor-Index -{}".format(e))
            socket.close()
            self.logger.info("Closed socket for Client-{}:{}".format(client.id,client.ip))

        # status = {"success": 1} or {"success": 0}  for failure, try second time
        if status.get("success", False) and status["success"] == 1:
            return 1

        return 0


    def message_handler_poller(self):
        """
        Monitor: Status Poller
        Collect the status from all the clients and add status to status_queue
        :return:
        """

        context = zmq.Context()

        for client in self.clients:
            # print("Client IP: {}, PORT Index-{}, PORT Status-{}".format(client.ip, client.port_index, client.port_status))
            client.socket_status = context.socket(zmq.PULL)
            #print("INFO: Monitor-Poller, Connecting to client-app-{} tcp://{}:{}".format(client.id, client.ip, client.port_status))
            self.logger.info("Monitor-Poller Connecting to client-app-{} tcp://{}:{}".format(client.id, client.ip, client.port_status))
            client.socket_status.connect("tcp://{}:{}".format(client.ip, client.port_status))

        while True:

            ## Condition to break the loop:
            # - Received a status from all clients that all the status have been processed.
            # - Received a signal from master to shutdown
            self.status_lock.acquire()
            stop = self.stop_status_poller.value
            self.status_lock.release()

            if stop == 1:
                break

            previous_client_operation_status = 1
            # Send index data to each client in round-robin fashion
            for client in self.clients:
                # Use to check that data has been reached to client. Otherwise resend same data.
                #print("++++++++++++>>INFO: POLLER - Waiting to receive data from client-{}".format(client.id))
                try:
                    received_response = client.socket_status.poll(timeout=1000)  # Wait 1 secs
                    if received_response:
                        status = client.socket_status.recv_json()
                        self.logger.debug("Monitor-Poller Operation Status for client-{}, Status - {}".format(client.id, status))
                        self.status_queue.put(status)
                except Exception as e:
                    self.logger.excep("Monitor-Status-Poller {} ".format(e))


        # Closing all socket connection
        try:
            for client in self.clients:
                client.socket_status.close()
            # Terminate context.
            context.term()
        except Exception as e:
            self.logger.excep("{}: minitor-poller, {}".format(e))

        self.status_lock.acquire()
        self.monitor_status_poller.value = 1
        self.status_lock.release()
        #print("INFO: Monitor-Status-Poller terminated gracefully! ")
        self.logger.info("Monitor-Status-Poller terminated gracefully! ")



    def operation_progress(self):
        """
        Monitor: Operation Progress
        Process the status result coming from different client nodes and update the progress.
        #TODO
        :return:
        """
        file_index_count = 0
        operation_success_count = 0
        operation_failure_count = 0
        display_percentage = 10
        failure_file_size_in_byte = 0
        while True:
            # Condition to break the loop:
            # If status poller has stopped  and status_queue is empty

            # Forcefully stop the process
            self.status_lock.acquire()
            status = self.stop_status_poller.value
            self.status_lock.release()

            if status == 1 and self.status_queue.empty():
                break


            if not self.status_queue.empty():
                status = self.status_queue.get()
                operation_success_count += status.get("success", 0)
                if status.get("failure", 0):
                    operation_failure_count += status.get("failure", 0)
                    failure_file_size_in_byte += status.get("size", 0)

                # Progress calculation, the value of index_data_count increases as indexing at progresses.
                # The file_index_count increases as response received from client application.
                #print("*****DEBUG: Monitor-Progress operation progress status - Total Files-{}, Success-{}, Failure-{}".format(self.index_data_count.value, operation_success_count, operation_failure_count))
                file_index_count = operation_success_count + operation_failure_count
                #upload_parcentage = (file_index_count / self.index_data_count.value) * 100
                #print("*****INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_parcentage))
                #self.logger.info("*****INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_parcentage))

                if self.index_data_count.value:
                    upload_percentage = (file_index_count / self.index_data_count.value) * 100
                    if upload_percentage > display_percentage:
                        self.logger.info(" ** Monitor-Progress - Operation Status Progress - {:.2f}% **".format(upload_percentage))
                        display_percentage +=10

            """
            if self.all_index_data_distributed.value:
                if self.index_data_count.value:
                    upload_percentage = (file_index_count / self.index_data_count.value) * 100
                    if upload_percentage > display_percentage:
                        print("*****INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_percentage))
                        self.logger.info("INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_percentage))
                        display_percentage +=10
            """


            ## All index data distributed to clients and received all operation status back from clients.
            if self.all_index_data_distributed.value and  file_index_count ==  self.index_data_count.value:
                self.status_lock.acquire()
                self.stop_status_poller.value = 1   # This will stop Monitor-Poller
                self.status_lock.release()

        total_operation_time = (datetime.now() - self.operation_start_time).seconds

        # Calculate operation BandWidth
        if self.operation.upper() == "DEL" or self.operation.upper() == "GET":
            prefix_index_data_file = "/var/log/prefix_index_data.json"
            with open(prefix_index_data_file, "r") as prefix_index_data_handler:
                prefix_index_data = json.load(prefix_index_data_handler)
        else:
            prefix_index_data = self.prefix_index_data
        original_file_size_in_byte = 0
        for prefix,value in prefix_index_data.items():
            original_file_size_in_byte += value["size"]

        success_operation_size_in_byte= original_file_size_in_byte

        self.logger.info("***** Operation Statistics *****")
        if operation_success_count:
            success_percentage = (operation_success_count / self.index_data_count.value) * 100
            self.logger.info("Total Operation: {},  Operation Success:{} - {:.2f}%".format(self.index_data_count.value, operation_success_count, success_percentage))
        if operation_failure_count:
            failure_percentage = (operation_failure_count / self.index_data_count.value) * 100
            self.logger.info("Total Operation:{}, Operation Failure:{} - {:.2f}%".format(self.index_data_count.value, operation_failure_count,failure_percentage))
            success_operation_size_in_byte -= failure_file_size_in_byte

        bandwidth = 0
        if total_operation_time:
            bandwidth = success_operation_size_in_byte / ( 1024 * 1024 * total_operation_time)
        operation_size_in_gb = success_operation_size_in_byte / ( 1024 * 1024 * 1024)

        self.logger.info("Operation {} completed in {} seconds for {:.2f} GiB ".format(self.operation,  total_operation_time, operation_size_in_gb))
        self.logger.info("Operation {} BandWidth = {:.2f} MiB/sec ".format(self.operation, bandwidth))
        self.monitor_progress_status.value = 1
        self.logger.info("Monitor-Progress-Status terminated! ")
