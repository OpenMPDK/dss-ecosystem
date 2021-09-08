#!/usr/bin/python3
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
import socket
import json
import time
from datetime import datetime

# IP address family
IP_ADDRESS_FAMILY = {
    "IPv4": socket.AF_INET,
    "IPv6": socket.AF_INET6
}

CONNECTION_TIME_THRESHOLD = 180 # 3 Minutes, maximum wait time for socket connection.

class ClientSocket:
    def __init__(self, logger=None, sock=None, ip_address_family="IPv4"):
        self.logger = logger
        if sock is None:
            if ip_address_family.upper() == "IPV4":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            elif ip_address_family.upper() == "IPV6":
                self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            else:
                self.logger.error("Wrong ip_address_family - {}, Supported {}".format(ip_address_family,
                                                                                      IP_ADDRESS_FAMILY))
        else:
            self.socket = sock

    def connect(self,host,port):
        """
        Connect to a socket with the specified host and port.
        :param host:
        :param port:
        :return:
        """
        if not host:
            raise ConnectionError("Host not specified")
        if not port:
            raise ConnectionError("Port not specified")

        is_connection_refused = True
        connection_time_start = datetime.now()
        while is_connection_refused:
            is_connection_refused =False
            try:
                self.socket.connect((host, int(port)))
            except ConnectionRefusedError as e:
                self.logger.warn("ConnectionRefusedError - retrying to connect")
                is_connection_refused = True
            except ConnectionError as e:
                self.logger.excep("ConnectionError -- {}".format(e))
            except Exception as e:
                self.logger.excep("GenericException - {}".format(e))
            time.sleep(1)
            if (datetime.now() - connection_time_start).seconds > CONNECTION_TIME_THRESHOLD:
                raise TimeoutError("Client socket connection timedout!")

    def send_json(self,message={}):
        """
        Send a JSON formatted data.
        :param message: JSON/DICT
        :return: Success/Failure = Tue/False
        """
        if message:
            msg_body = json.dumps(message)
            msg_len = (str(len(msg_body))).zfill(10)
            msg = msg_len + msg_body
            try:
                # sendall on success return None.
                if self.socket.sendall(msg.encode("ascii") ) is None:
                    #self.logger.info("SENT MSG Len:{}, Size:{}".format(msg_len,len(msg_body)))
                    return True
                else:
                    self.logger.error("Failed to send msg - {}".format(msg))
            except BrokenPipeError as e:
                self.logger.excep("BrokenPipeError - {}".format(e))
            except Exception as e:
                self.logger.excep("Message Send Failed - {}".format(e))
        return False

    def recv_json(self):
        """
        Receive the data from socket based on data length. and return data in JSON format.
        :return: Return received data in json format.
        """
        msg_len = None
        msg = {}
        msg_body =None
        try:
            msg_len = self.socket.recv(10)
            if msg_len == b'':
                raise RuntimeError("Socket connection broken")
        except Exception as e:
            print("Exception: {}".format(e))

        try:
            msg_body = self.socket.recv(int(msg_len))
            if msg_body == b'':
                raise RuntimeError("Socket connection broken")
        except Exception as e:
            print("Exception: {}".format(e))

        if msg_body:
            msg = msg_body.decode("ascii")
        if format.upper() == "JSON":
            return json.loads(msg)
        else:
            return msg
    def close(self):
        try:
            #self.socket.shutdown(SHUT_RDWR)
            self.socket.close()
        except Exception as e:
            self.logger.excep("Close socket {}".format(e))


class ServerSocket:
    def __init__(self, logger=None, sock=None, ip_address_family="IPv4"):
        if sock is None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.socket = sock
        self.logger = logger

    def bind(self,host,port):
        if not host:
            self.logger.error("ERROR: Host not specified")
        if not port:
            self.logger.error("ERROR: Port not specified")
        port = int(port)
        try:
            self.socket.bind((host,port))
            self.socket.listen(5)
            self.logger.info("Client is listening message on {}:{} ".format(host,port))
        except ConnectionError as e:
            self.logger.error("Address ({}:{}) bind error - {}".format(e))
        except Exception as e:
            self.logger.error("Not able to bind to host={}:{}".format(host,port))

    def accept(self):
        client_sock, address = self.socket.accept()
        self.client_socket = client_sock
        self.logger.info("Connected to {}".format(str(address)))

    def send_json(self,message=None,format="JSON"):
        """
        Send data to a client
        :param message: STRING|DICT|JSON ,
        :param format: require to specify the format. By default JSON.
        :return:
        """
        if message:
            msg_body = message
            if format.upper() == "JSON":
                msg_body = json.dumps(message)

            msg_len = (str(len(msg_body))).zfill(10)
            msg = msg_len + msg_body

            try:
                # sendall on success return None.
                if self.client_socket.sendall(msg.encode("ascii") ) is None:
                    return True
                else:
                    self.logger.error("Failed to send msg - {}".format(msg_body))
            except RuntimeError as e:
                    self.logger.error("RuntimeError - {}".format(e))
            except Exception as e:
                raise RuntimeError("Message Send Failed")
        return False

    def recv_json(self, format="JSON"):
        """
        Receive the data from socket based on data length. and return data in JSON format.
        :return: Return received data in json format.
        """
        msg_len = None
        msg = {}
        try:
            msg_len = self.client_socket.recv(10)
            #self.logger.info("MSG Length: {}".format(msg_len))
            if msg_len == b'':
                raise RuntimeError("Socket connection broken")
            msg_len = int(msg_len)
        except Exception as e:
            self.logger.error("Determine msg length - {}".format(e))
        if msg_len:
            msg_body = b''
            try:
                # Iterate till we receive desired number of bytes.
                received_data_size = 0
                while received_data_size < msg_len:
                    data_size = msg_len - received_data_size
                    msg_body += self.client_socket.recv(data_size)
                    received_data_size += len(msg_body)

                if msg_body == b'':
                    raise RuntimeError("Socket connection broken")
                #self.logger.info("RECEIVED MSG-LEN:{}, SIZE:{}".format(msg_len, len(msg_body)))
            except Exception as e:
                self.logger.excep("ServerSocket receive bytes -  {}".format(e))
            if msg_body :
                msg = msg_body.decode("ascii")
            if format.upper() == "JSON":
                return json.loads(msg)
            else:
                return msg

    def close(self):
        """
        Close server side socket.
        :return:
        """
        try:
            self.socket.close()
        except Exception as e:
            self.logger.error("Closing Socket - {}".format(e))




