#!/usr/bin/python

import os,sys
import time
import  zmq
from multiprocessing import Process,Queue,Value, Lock
from datetime import datetime
import json

"""
Monitor the progress of operation.
- Communicate to client , poll the status periodically.
- Process result 
- Display/Store result
"""


class Monitor:

    def  __init__(self, clients, index_data_queue, index_data_lock, index_data_generation_complete, logger_queue, logger_lock, operation):
        self.clients = clients
        self.index_data_queue=index_data_queue
        self.index_data_lock = index_data_lock
        self.index_data_generation_complete = index_data_generation_complete
        self.operation = operation
        self.logger_queue = logger_queue
        self.logger_lock  = logger_lock

        self.status_queue = Queue()
        self.status_lock  = Lock()
        self.stop_status_poller = Value('i', 0)
        self.index_data_count = Value('i', 0)

        self.all_index_data_distributed = Value('i', 0)

        # Monitor termination
        self.monitor_index_data_sender = Value('i', 0)
        self.monitor_status_poller = Value('i', 0)
        self.monitor_progress_status = Value('i', 0)

        self.operation_start_time = datetime.now()



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

        print("DEBUG: Stopping MessageHandler - ")
        self.status_lock.acquire()
        self.stop_status_poller.value = 1
        self.status_lock.release()
        time.sleep(2)

        while self.process_index.is_alive():
            time.sleep(1)
            try:
                self.process_index.terminate()
            except Exception as e:
                print("EXCEPTION: Unable to terminate MessageHandler - IndexSender {}".format(e))

        while self.process_status.is_alive():
            time.sleep(1)
            try:
                self.process_status.terminate()
            except Exception as e:
                print("EXCEPTION: Unable to terminate MessageHandler - StatusPoller ...\n {}".format(e))

        self.logger_queue.put("INFO: Stopped all monitor processes ... ")
        print("INFO: All monitor stopped ...")


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
            print("INFO: Connecting to INDEX MessageHandler tcp://{}:{}".format(client.ip, client.port_index))
            self.logger_queue.put("INFO: Connecting to INDEX MessageHandler tcp://{}:{}".format(client.ip, client.port_index))
            client.socket_index.connect("tcp://{}:{}".format(client.ip, client.port_index))

        # Buffer to store prefix index and file count  {"prefix":<file count>}
        prefix_index_data = {}

        while True:

            # Forcefully stop the process
            self.status_lock.acquire()
            stop = self.stop_status_poller.value
            self.status_lock.release()
            if stop :
                self.logger_queue.put("ERROR: Forcefully shutting down Monitor-index!")
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
                #print("YYYYYYYYYYYYY: New_list Data: Client-{}, MSG-{}".format(client.id, data))
                # Send data to ClientApplication running on a  Client-Physical Node
                if data:
                    #print("FFFFFFFFFFFFFFF:client{}:{}".format(client.id, data))
                    # Buffer prefix_index_data for persistent storage only to be used during PUT
                    if self.operation.upper() == "PUT":
                        object_prefix_key = data["dir"][1:] + "/"
                        if data["dir"] in prefix_index_data:
                            prefix_index_data[object_prefix_key] += len(data["files"])
                        else:
                            prefix_index_data[object_prefix_key] = len(data["files"])

                    #print("===>>INFO: Sending index data - {}:{} -> {}".format(client.ip, client.port_index, data))
                    if self.send_index_data(client.socket_index, data):
                        self.index_data_count.value += len(data.get("files", []))
                        previous_client_operation_status = 1
                    else: # Re-send once again
                        previous_client_operation_status = 0
                        if self.send_index_data(client.socket_index, data):
                            self.index_data_count.value += len(data.get("files", []))
                            previous_client_operation_status = 1

            if self.index_data_generation_complete.value == 1  and self.index_data_queue.empty():
                self.logger_queue.put("INFO: Indexed data distribution is completed!")
                self.all_index_data_distributed.value = 1
                # Inform all the client applications running on different nodes
                for client in self.clients:
                    data = {"indexing_done": 1}
                    if not self.send_index_data(client.socket_index, data):
                        if not self.send_index_data(client.socket_index, data):
                            self.logger_queue.put("ERROR: Unable to send indexing completion message to client-{}".format(client.id))
                    print("INFO: Indexed data distribution is completed, Notifying client application {}:{} -> {}".format(client.ip, client.port_index, data))
                break

        # Closing all socket connection
        try:
            for client in self.clients:
                client.socket_index.close()
        except Exception as e:
            self.logger_queue.put("EXCEPTION:{}: monitor-index, {}".format(e))

        # Update MasterApplication about termination of Monitor
        self.monitor_index_data_sender.value = 1

        # Storing prefix index data to persistent storage
        if self.operation.upper() == "PUT":
            prefix_storage_file = "/var/log/prefix_index_data.json"
            print("INFO: Storing prefix_index_data to persistent storage - {}".format(prefix_storage_file))
            with open(prefix_storage_file, "w") as persistent_storage:
                json.dump(prefix_index_data, persistent_storage)

        print("INFO: Monitor-Index-Distribution is terminated gracefully!")
        self.logger_queue.put("INFO: Monitor-Index-Distribution is terminated gracefully! ")



    def send_index_data(self, socket, data):
        """
        Send index data for a socket client. On failure, resend once.
        :param socket: client socket
        :param data: index data
        :return: success/failure 1/0
        """
        status = {}
        try:
            socket.send_json(data) # Send index data
            # Wait (3sec) for ClientApplication's response on operation. Otherwise re-send index data.
            received_response = socket.poll(timeout=2000)  # Wait 3 secs
            if received_response:
                status = socket.recv_json()
            else:
                print("WARNING: Monitor-Index -- RSP not received from ClientApp - {}".format(status))
        except Exception as e:
            self.logger_queue.put("EXCEPTION: Monitor-Index -{}".format(e))
            print("EXCEPTION: Monitor-Index -{}".format(e))

        # status = {"success": 1} or {"success": 0}  for failure, try second time
        if status.get("success", False) and status["success"] == 1:
            return 1

        print("ERROR: Monitor-INDEX -- ClientApp response - {}".format(status))
        return 0


    def message_handler_poller(self):
        """
        Monitor: Status Poller
        Collect the status from all the clients and add status to status_queue
        :return:
        """
        #print("INFO: Poller Started ....")
        context = zmq.Context()

        for client in self.clients:
            # print("Client IP: {}, PORT Index-{}, PORT Status-{}".format(client.ip, client.port_index, client.port_status))
            client.socket_status = context.socket(zmq.PULL)
            print("INFO: Monitor-Poller, Connecting to client-app tcp://{}:{}".format(client.ip, client.port_status))
            self.logger_queue.put("DEBUG: Monitor-Poller Connecting to client-app STATUS MsgHandler tcp://{}:{}".format(client.ip, client.port_status))
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
                        #print("++++DEBUG: Monitor-Poller Operation Status client-{}, Status - {}".format(client.id, status))
                        self.logger_queue.put("DEBUG: Monitor-Poller Operation Status for client-{}, Status - {}".format(client.id, status))
                        self.status_queue.put(status)
                except Exception as e:
                    print("EXCEPTION: Monitor-Status-Poller {} ".format(e))

        # Closing all socket connection
        try:
            for client in self.clients:
                client.socket_status.close()
        except Exception as e:
            self.logger_queue.put("EXCEPTION:{}: minitor-poller, {}".format(e))

        self.status_lock.acquire()
        self.monitor_status_poller.value = 1
        self.status_lock.release()
        print("INFO: Monitor-Status-Poller terminated gracefully! ")
        self.logger_queue.put("INFO: Monitor-Status-Poller terminated gracefully! ")



    def operation_progress(self):
        """
        Monitor: Operation Progress
        Process the status result and update the progress.
        #TODO
        :return:
        """
        file_index_count = 0
        operation_success_count = 0
        operation_failure_count = 0
        display_percentage = 10
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
                operation_failure_count += status.get("failure", 0)

                # Progress calculation, the value of index_data_count increases as indexing at progresses.
                # The file_index_count increases as response received from client application.
                #print("*****DEBUG: Monitor-Progress operation progress status - Total Files-{}, Success-{}, Failure-{}".format(self.index_data_count.value, operation_success_count, operation_failure_count))
                file_index_count = operation_success_count + operation_failure_count
                #upload_parcentage = (file_index_count / self.index_data_count.value) * 100
                #print("*****INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_parcentage))
                #self.logger_queue.put("*****INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_parcentage))

            if self.all_index_data_distributed.value:
                if self.index_data_count.value:
                    upload_percentage = (file_index_count / self.index_data_count.value) * 100
                    if upload_percentage > display_percentage:
                        print("*****INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_percentage))
                        self.logger_queue.put("INFO: Monitor-Progress - Operation Status Progress - {:.2f}%".format(upload_percentage))
                        display_percentage +=10

            ## All index data distributed to clients and received all operation status back from clients.
            if self.all_index_data_distributed.value and  file_index_count ==  self.index_data_count.value:
                self.status_lock.acquire()
                self.stop_status_poller.value = 1   # This will stop Monitor-Poller
                self.status_lock.release()


        self.monitor_progress_status.value = 1

        print("***** Operation Statistics *****")
        if operation_success_count:
            success_percentage = (operation_success_count / self.index_data_count.value) * 100
            print("Total Operation: {},  Operation Success:{} - {}%".format(self.index_data_count.value, operation_success_count, success_percentage))
        if operation_failure_count:
            failure_percentage = (operation_failure_count / self.index_data_count.value) * 100
            print("Total Operation:{}, Operation Failure:{} - {}%".format(self.index_data_count.value, operation_failure_count,failure_percentage))
        now = datetime.now()
        print("INFO: Operation {} completed in {} seconds ".format(self.operation, (now - self.operation_start_time).seconds))
        print("INFO: Monitor-Progress-Status terminated!")
        self.logger_queue.put("INFO: Monitor-Progress-Status terminated! ")
