#!/usr/bin/python3
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

import socket
import json
import time
from datetime import datetime

# IP address family
IP_ADDRESS_FAMILY = {
    "IPv4": socket.AF_INET,
    "IPv6": socket.AF_INET6
}

CONNECTION_DELAY_INTERVAL = 2
CONNECTION_TIME_THRESHOLD = 300  # 5 Minutes, maximum wait time for socket connection.
MESSAGE_LENGTH = 10  # Message length 10 bytes
RECV_TIMEOUT = 60  # Wait to receive data from socket for 60 seconds.


class ClientSocket:

    def __init__(self, logger=None, ip_address_family="IPv4"):
        self.logger = logger
        self.socket = None
        if ip_address_family.upper() == "IPV4":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif ip_address_family.upper() == "IPV6":
            self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            self.logger.error("Wrong ip_address_family - {}, Supported {}".format(ip_address_family, IP_ADDRESS_FAMILY))
            raise ConnectionError("Socket initialization failed! ")

    def connect(self, host, port):
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
        time_to_sleep = CONNECTION_DELAY_INTERVAL
        while is_connection_refused:
            is_connection_refused = False
            try:
                self.socket.connect((host, int(port)))
            except ConnectionRefusedError as e:
                self.logger.warn(f"ConnectionRefusedError - retrying to connect {time_to_sleep}")
                is_connection_refused = True
            except ConnectionError as e:
                self.logger.excep(f"{host}:{port}-ConnectionError - {e}")
            except socket.error as e:
                self.logger.excep(f"{host}:{port}-GenericException - {e}")
            except Exception as e:
                self.logger.error(f"{host}:{port}->{e}")
            # Add a delay in increasing order of 2 sec.
            time.sleep(time_to_sleep)
            time_to_sleep += CONNECTION_DELAY_INTERVAL
            if (datetime.now() - connection_time_start).seconds > CONNECTION_TIME_THRESHOLD:
                raise socket.timeout("Socket connection timeout=300sec !")

    def send_json(self, message={}):
        """
        Send a JSON formatted data.
        :param message: JSON/DICT
        :return: Success/Failure = Tue/False
        """
        if message and self.socket:
            msg_body = json.dumps(message)
            msg_len = (str(len(msg_body))).zfill(MESSAGE_LENGTH)
            if not msg_body.startswith('{') or not msg_body.endswith('}'):
                self.logger.error("ClientSocket: BAD MSG - {}".format(msg_body))
            msg = msg_len + msg_body

            try:
                # sendall on success return None.
                if self.socket.sendall(msg.encode("utf8", "ignore")) is None:
                    return True
                else:
                    self.logger.error("Failed to send msg - {}".format(msg))
            except BrokenPipeError as e:
                self.logger.excep("BrokenPipeError - {}".format(e))
            except ConnectionError as e:
                self.logger.error("ConnectionError- {}".format(e))
            except MemoryError as e:
                self.logger.error("MemoryError- The mesage size is more than supported on this system. {}".format(e))
            except socket.timeout as e:
                raise socket.timeout()
            except socket.error as e:
                self.logger.excep("Message Send Failed - {}".format(e))
        return False

    def recv_json(self, format="JSON", timeout=RECV_TIMEOUT):
        """
        Receive the data from socket based on data length. and return data in JSON format.
        This is blocking call and wait for data from socket end utill received desired
        number of bytes.
        The message contains 10 bytes header and body. Read header untill received 10 bytes or timeout.
        Iterate to receive desired number of bytes from socket.
        :format: JSON/String
        :timeout: default 60 seconds
        :return: Return received data in json format.
        """
        msg_len = None
        msg = "{}"

        try:
            msg_len_in_bytes = b''
            received_msg_len_size = 0
            time_started = datetime.now()
            # Iterate till we receive 10 bytes or timeout.
            while received_msg_len_size < MESSAGE_LENGTH:
                received_msg_len_in_bytes = self.socket.recv(MESSAGE_LENGTH - received_msg_len_size)
                received_msg_len_size += len(received_msg_len_in_bytes)
                msg_len_in_bytes += received_msg_len_in_bytes
                time_spent_in_seconds = (datetime.now() - time_started).seconds
                if time_spent_in_seconds >= timeout:
                    raise socket.timeout("ClientSocket: Timeout ({} seconds) from recv function".format(
                        time_spent_in_seconds))
            if msg_len_in_bytes != b'':
                msg_len = int(msg_len_in_bytes)
        except socket.timeout as e:
            raise e
        except socket.error as e:
            raise socket.error("ClientSocket: Incorrect message length - {}".format(e))
        except ValueError as e:
            raise ValueError("ClientSocket: ValueError - {}".format(e))

        if msg_len:
            msg_body = b''
            # Iterate till we receive desired number of bytes.
            received_data_size = 0
            while received_data_size < msg_len:
                data_size = msg_len - received_data_size
                try:
                    received_data = self.socket.recv(data_size)
                    received_data_size += len(received_data)
                    msg_body += received_data
                except socket.timeout as e:
                    self.logger.error("ClientSocket: Timeout-{}".format(e))
                    break
                except socket.error as e:
                    self.logger.error("ClientSocket: {}".format(e))

            if msg_body == b'':
                raise RuntimeError("ClientSocket: Empty message for message length -{}".format(msg_len))

            if msg_body:
                if len(msg_body) == msg_len:
                    msg = msg_body.decode("utf8", "ignore")
                else:
                    self.logger.error("ClientSocket: Received incomplete message.")
        if format.upper() == "JSON":
            json_data = {}
            try:
                json_data = json.loads(msg)
            except json.JSONDecodeError as e:
                self.logger.error("ClientSocket: Bad JSON data - {},{}, {}".format(msg_len, msg, e))
            except Exception as e:
                raise Exception("Bad formed message - {}{}, error- {}".format(msg_len, msg, e))

            return json_data
        else:
            return msg

    def close(self):
        """
        Close client socket
        :return:
        """
        try:
            # self.socket.shutdown()
            self.socket.close()
        except Exception as e:
            self.logger.excep("Close socket {}".format(e))


class ServerSocket:

    def __init__(self, logger=None, ip_address_family="IPv4"):
        self.socket = None
        self.logger = logger
        self.client_socket = None
        if ip_address_family.upper() == "IPV4":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif ip_address_family.upper() == "IPV6":
            self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            self.logger.error("Wrong ip_address_family - {}, Supported {}".format(ip_address_family, IP_ADDRESS_FAMILY))
            raise ConnectionError("Socket initialization failed!")

    def bind(self, host, port):
        """
        Bind server socket
        :param host:
        :param port:
        :return:
        """
        if not host:
            self.logger.error("ERROR: Host not specified")
        if not port:
            self.logger.error("ERROR: Port not specified")
        port = int(port)
        try:
            self.socket.bind((host, port))
            self.socket.listen(5)
            self.logger.info("Client is listening for message on {}:{} ".format(host, port))
        except ConnectionError as e:
            self.logger.error("Address ({}:{}) bind error - {}".format(host, port, e))
        except Exception as e:
            self.logger.error("Not able to bind to host={}:{}, {}".format(host, port, e))

    def accept(self):
        """
        Accept client request.
        :return:
        """
        self.client_socket, address = self.socket.accept()
        self.logger.info("Connected to {}".format(str(address)))

    def send_json(self, message=None, format="JSON"):
        """
        Send data to a client
        :param message: STRING|DICT|JSON ,
        :param format: require to specify the format. By default JSON.
        :return: Success/failure (True/False)
        """
        if message and self.client_socket:
            msg_body = message
            if format.upper() == "JSON":
                msg_body = json.dumps(message)
            if not msg_body.startswith('{') or not msg_body.endswith('}'):
                self.logger.error("ServerSocket: BAD MSG - {}".format(msg_body))

            msg_len = (str(len(msg_body))).zfill(MESSAGE_LENGTH)
            msg = msg_len + msg_body
            try:
                # sendall on success return None.
                if self.client_socket.sendall(msg.encode("utf8", "ignore")) is None:
                    return True
                else:
                    self.logger.error("Failed to send msg - {}".format(msg_body))
            except BrokenPipeError as e:
                self.logger.error("BrokenPipeError- Socket connection closed by the peer! {}".format(e))
            except ConnectionError as e:
                self.logger.error("ConnectionError- {}".format(e))
            except MemoryError as e:
                self.logger.error("MemoryError- The message size is more than supported on this system. {}".format(e))
            except RuntimeError as e:
                self.logger.error("RuntimeError - {}".format(e))
            except socket.error as e:
                self.logger.error("Message Send Failed - {}".fromat(e))
        return False

    def recv_json(self, format="JSON", timeout=RECV_TIMEOUT):
        """
        Receive the data from socket based on data length. and return data in JSON format.
        This is blocking call and wait for data from socket end utill received desired
        number of bytes.
        The message contains 10 bytes header and body. Read header untill received 10 bytes or timeout.
        Iterate to receive desired number of bytes from socket.
        :format: JSON/String
        :timeout: default 60 seconds
        :return: Return received data in json format.
        """
        msg_len = None
        msg = "{}"
        try:
            msg_len_in_bytes = b''
            received_msg_len_size = 0
            time_started = datetime.now()
            # Iterate till we receive 10 bytes or timeout.
            while received_msg_len_size < MESSAGE_LENGTH:
                received_msg_len_in_bytes = self.client_socket.recv(MESSAGE_LENGTH - received_msg_len_size)
                received_msg_len_size += len(received_msg_len_in_bytes)
                msg_len_in_bytes += received_msg_len_in_bytes
                time_spent_in_seconds = (datetime.now() - time_started).seconds
                if time_spent_in_seconds >= timeout:
                    raise socket.timeout("ServerSocket: Timeout ({} seconds) from recv function".format(time_spent_in_seconds))

            if msg_len_in_bytes != b'':
                msg_len = int(msg_len_in_bytes.decode('utf8'))
        except socket.timeout as e:
            raise e
        except socket.error as e:
            self.logger.error("ServerSocket: Determine msg length - {}".format(e))
        except ValueError as e:
            raise socket.error("ServerSocket: ValueError - {}".format(e))
        except Exception as e:
            self.logger.error("ServerSocket: {}".format(e))

        if msg_len:
            msg_body = b''
            received_data_size = 0
            # Iterate till we receive desired number of bytes.
            while received_data_size < msg_len:
                data_size = msg_len - received_data_size
                try:
                    received_data = self.client_socket.recv(data_size)
                    received_data_size += len(received_data)
                    msg_body += received_data
                except socket.timeout as e:
                    self.logger.error("ServerSocket: Timeout - {}".format(e))
                except socket.error as e:
                    self.logger.excep("ServerSocket receive bytes -  {}".format(e))
            if msg_body == b'':
                raise RuntimeError("Empty message for message length -{}".format(msg_len))

            if msg_body:
                if len(msg_body) == msg_len:
                    msg = msg_body.decode("utf8", "ignore")
                else:
                    self.logger.error("ServerSocket: Received incomplete message.")
        if format.upper() == "JSON":
            json_data = {}
            try:
                json_data = json.loads(msg)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError("ClientSocket: Bad JSON data - {}".format(e))
            except MemoryError as e:
                raise MemoryError("MemoryError: JSON load failed - {}".format(e))
            except Exception as e:
                raise Exception("ClientSocket: Bad JSON data - {}, error- {}".format(msg, e))

            return json_data
        else:
            return msg

    def close(self):
        """
        Close server side socket.
        :return: None
        """
        try:
            self.client_socket.close()
            time.sleep(1)
            self.socket.close()
        except Exception as e:
            self.logger.error("Closing Socket - {}".format(e))
