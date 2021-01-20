#!/usr/bin/python

import os,sys
import time
import  zmq
from multiprocessing import Process,Queue,Value, Lock


"""
Monitor the progress of operation.
- Communicate to client , poll the status periodically.
- Process result 
- Display/Store result
"""


class Monitor:

    def  __init__(self, clients, index_data_queue, index_data_lock, index_data_generation_complete, logger_queue, logger_lock):
        self.clients = clients
        self.index_data_queue=index_data_queue
        self.index_data_lock = index_data_lock
        self.index_data_generation_complete = index_data_generation_complete

        self.logger_queue = logger_queue
        self.logger_lock  = logger_lock

        self.status_queue = Queue()
        self.status_lock  = Lock()
        self.stop_status_poller = Value('i', 0)


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
        It stop processing when either all index is processed and thus "index_data_generation_complete" is set to 1
        or forcefully stopped through "stop_status_poller" by setting it to 1.
        :return:
        """

        context = zmq.Context()

        for client in self.clients:
            client.socket_index = context.socket(zmq.REQ)
            print("INFO:Connecting to INDEX MessageHandler tcp://{}:{}".format(client.ip, client.port_index))
            client.socket_index.connect("tcp://{}:{}".format(client.ip, client.port_index))

        while True:
            # Condition to break the loop:
            # - Check all index data is processed.
            # - Shutdown of application - forcefully.

            # Forcefully stop the process
            self.status_lock.acquire()
            stop = self.stop_status_poller.value
            self.status_lock.release()

            self.index_data_lock.acquire()
            indexing_completed = self.index_data_generation_complete.value
            self.index_data_lock.release()

            if stop or indexing_completed == 1 :
                break

            previous_client_operation_status = 1
            # Send index data to each client in round-robin fashion
            for client in self.clients:

                data = {}
                # Get data from shared index queue, if the previous client processed data successfully
                if previous_client_operation_status:
                    self.index_data_lock.acquire()
                    if not self.index_data_queue.empty():
                        data = self.index_data_queue.get()
                    self.index_data_lock.release()
                if data:
                    # Send index data
                    client.socket_index.send_json(data)
                    print("===>>INFO: Sending index data - {}:{} -> {}".format(client.ip,client.port_index, data))

                    # Use to check that data has been reached to client. Otherwise resend same data.
                    status = client.socket_index.recv_json()
                    #print("===>>: INDEX -- Received data - {}".format(status))

                    # status = {"success": 1} or {"success": 0}  for failure, try second time
                    if status.get("success", False) and status["success"] == 1:
                        previous_client_operation_status = 1
                        print("===>>>INFO: INDEX Send/Recv is good")
                    else:
                        # Re-try for sending data once.
                        client.socket_index.send_json(data)
                        status = client.socket_index.recv_json()

                        if status["success"] == 0:
                            previous_client_operation_status = 0

        self.logger_queue.put("Closing Monitor - MessageHandler -Index")
        # Closing all socket connection
        for client in self.clients:
            client.socket_index.close()

    def message_handler_poller(self):
        """
        Collect the status from all the clients and add status to status_queue
        :return:
        """
        print("INFO: Poller Started ....")
        context = zmq.Context()

        for client in self.clients:
            # print("Client IP: {}, PORT Index-{}, PORT Status-{}".format(client.ip, client.port_index, client.port_status))
            client.socket_status = context.socket(zmq.PULL)
            print("Connecting to STATUS MessageHandler tcp://{}:{}".format(client.ip, client.port_status))
            client.socket_status.connect("tcp://{}:{}".format(client.ip, client.port_status))

        while True:

            ## Condition to break the loop:
            # - Received a status from all clients that all the status have been processed.
            # - Received a signal from master to shutdown

            # Forcefully stop the process
            self.status_lock.acquire()
            stop = self.stop_status_poller.value
            self.status_lock.release()

            if stop == 1:
                break

            previous_client_operation_status = 1
            # Send index data to each client in round-robin fashion
            for client in self.clients:
                # Use to check that data has been reached to client. Otherwise resend same data.
                print("++++++++++++>>INFO: POLLER - Waiting to receive data from client-{}".format(client.id))
                status = client.socket_status.recv_json()
                print("+++++++++>>MSG Handler: Status - {}".format(status))
                self.status_lock.acquire()
                self.status_queue.put(status)
                self.status_lock.release()
        self.logger_queue.put("Closing Monitor - MessageHandler -Status")

        # Closing all socket connection
        for client in self.clients:
            client.socket_status.close()



    def operation_progress(self):
        """
        Process the status result and update the progress.
        #TODO
        :return:
        """

        while True:
            # Condition to break the loop:
            # If status poller has stopped  and status_queue is empty

            # Forcefully stop the process
            self.status_lock.acquire()
            status = self.stop_status_poller.value
            is_status_queue_empty =  self.status_queue.empty()
            self.status_lock.release()
            if status == 1 and is_status_queue_empty:
                break


            if not is_status_queue_empty:
                status = self.status_queue.get()
                print("*****>>Operation Status: {}".format(status))

                # Process status means display
                # - Check total file index
                # - Add status count
                # - Make a difference and return result.




if __name__ == "__main__":
    monitor = Monitor()