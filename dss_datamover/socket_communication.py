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

DEFAULT_CONNECTION_DELAY_INTERVAL = 2
DEFAULT_CONNECTION_TIME_THRESHOLD = 300  # 5 Minutes, maximum wait time for socket connection.
DEFAULT_MESSAGE_LENGTH = 10  # Message length 10 bytes
DEFAULT_RECV_TIMEOUT = 60  # Wait to receive data from socket for 60 seconds.


class ClientSocket:

    def __init__(self, config, logger=None, ip_address_family="IPv4"):
        self.logger = logger
        self.config = config
        self.socket = None

        self.CONNECTION_DELAY_INTERVAL = (
            self.config.get("socket_options", DEFAULT_CONNECTION_DELAY_INTERVAL).get("connection_delay_interval", DEFAULT_CONNECTION_DELAY_INTERVAL)
        )
        self.CONNECTION_TIME_THRESHOLD = (
            self.config.get("socket_options", DEFAULT_CONNECTION_TIME_THRESHOLD).get("connection_time_threshold", DEFAULT_CONNECTION_TIME_THRESHOLD)
        )
        self.MESSAGE_LENGTH = (
            self.config.get("socket_options", DEFAULT_MESSAGE_LENGTH).get("message_length", DEFAULT_MESSAGE_LENGTH)
        )
        self.RECV_TIMEOUT = (
            self.config.get("socket_options", DEFAULT_RECV_TIMEOUT).get("recv_timeout", DEFAULT_RECV_TIMEOUT)
        )

        if ip_address_family.upper() == "IPV4":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif ip_address_family.upper() == "IPV6":
            self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            self.logger.error("Wrong ip_address_family - {}, Supported {}".format(ip_address_family, IP_ADDRESS_FAMILY))
            raise ConnectionError("Socket initialization failed! ")
        # configures socket to send data as soon as it is available, regardless of packet size
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

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
        time_to_sleep = self.config.get("socket_options", ).get("connection_delay_interval")
        time_to_sleep = self.CONNECTION_DELAY_INTERVAL
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
            time_to_sleep += self.CONNECTION_DELAY_INTERVAL
            if (datetime.now() - connection_time_start).seconds > self.CONNECTION_TIME_THRESHOLD:
                raise socket.timeout("Socket connection timeout=300sec !")

    def send_json(self, message={}):
        """
        Send a JSON formatted data.
        :param message: JSON/DICT
        :return: Success/Failure = Tue/False
        """
        if message and self.socket:
            msg_body = json.dumps(message)
            msg_len = (str(len(msg_body))).zfill(self.MESSAGE_LENGTH)
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

    def recv_json(self, format="JSON"):
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
        self.socket.settimeout(self.RECV_TIMEOUT)
        msg = "{}"
        msg_len = 0
        time_started = datetime.now()

        try:
            # first receive message length from payload
            msg_len_in_bytes = b''
            msg_len_in_bytes = self.socket.recv(self.MESSAGE_LENGTH)
            msg_len = int(msg_len_in_bytes)
            if len(msg_len_in_bytes) != self.MESSAGE_LENGTH:
                raise RuntimeError(f"ClientSocket: Received incorrect message length header {msg_len} bytes")

            # now retrieve rest of payload from buffer based on msg_len
            # TODO: handle rare edge case where only partial bytes were read before socket termination
            msg_body = b''
            received_data_size = 0
            while received_data_size < msg_len:
                data_size = msg_len - received_data_size
                received_data = self.socket.recv(data_size)
                received_data_size += len(received_data)
                msg_body += received_data

            # process received payload
            if msg_body == b'':
                raise RuntimeError("ClientSocket: Empty message for message length -{}".format(msg_len))
            elif msg_body:
                if received_data_size == msg_len:
                    msg = msg_body.decode("utf8", "ignore")
                else:
                    raise RuntimeError("ClientSocket: Received incomplete message.")

            # return response as JSON
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
        except socket.timeout as e:
            self.logger.error("ClientSocket: Timeout ({} seconds) from recv function".format((datetime.now() - time_started).seconds))
            # return status depending on received message size, if incomplete a Runtime Error should have been raised
            if len(msg_len_in_bytes) == self.MESSAGE_LENGTH and received_data_size == msg_len:
                return json.loads(msg)
            else:
                return json.loads("{}")
        except socket.error as e:
            self.logger.error(f"ClientSocket SocketError: {e}")
            raise socket.error(f"ClientSocket SocketError: {e}")
        except ValueError as e:
            raise ValueError(f"ClientSocket: ValueError - {e}")
        except RuntimeError as e:
            self.logger.error(f"ClientSocket: RuntimeError {e}")
            raise RuntimeError(f"ClientSocket: RuntimeError {e}")
        except Exception as e:
            raise Exception(f"ClientSocket: Exception {e}")

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

    def __init__(self, config, logger=None, ip_address_family="IPv4"):
        self.socket = None
        self.logger = logger
        self.config = config
        self.client_socket = None

        self.CONNECTION_DELAY_INTERVAL = (
            self.config.get("socket_options", DEFAULT_CONNECTION_DELAY_INTERVAL).get("connection_delay_interval", DEFAULT_CONNECTION_DELAY_INTERVAL)
        )
        self.CONNECTION_TIME_THRESHOLD = (
            self.config.get("socket_options", DEFAULT_CONNECTION_TIME_THRESHOLD).get("connection_time_threshold", DEFAULT_CONNECTION_TIME_THRESHOLD)
        )
        self.MESSAGE_LENGTH = (
            self.config.get("socket_options", DEFAULT_MESSAGE_LENGTH).get("message_length", DEFAULT_MESSAGE_LENGTH)
        )
        self.RECV_TIMEOUT = (
            self.config.get("socket_options", DEFAULT_RECV_TIMEOUT).get("recv_timeout", DEFAULT_RECV_TIMEOUT)
        )

        if ip_address_family.upper() == "IPV4":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif ip_address_family.upper() == "IPV6":
            self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        else:
            self.logger.error("Wrong ip_address_family - {}, Supported {}".format(ip_address_family, IP_ADDRESS_FAMILY))
            raise ConnectionError("Socket initialization failed!")
        print(f"**Initialized Server socket here are the configs: {self.config}")

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
            """
            Previous execution may have left the socket in a TIME_WAIT state and can't be
            immediately reused. On POSIX platforms the SO_REUSEADDR socket option is set in order to immediately
            reuse previous sockets which were bound on the same address and remained in TIME_WAIT state.
            """
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

            msg_len = (str(len(msg_body))).zfill(self.MESSAGE_LENGTH)
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
                self.logger.error("Message Send Failed - {}".format(e))
        return False

    def recv_json(self, format="JSON"):
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
        self.client_socket.settimeout(self.RECV_TIMEOUT)
        msg_len = None
        msg = "{}"
        time_started = datetime.now()

        try:
            # first receive message length from payload
            msg_len_in_bytes = b''
            msg_len_in_bytes = self.client_socket.recv(self.MESSAGE_LENGTH)
            msg_len = int(msg_len_in_bytes.decode('utf8'))
            if len(msg_len_in_bytes) != self.MESSAGE_LENGTH:
                raise RuntimeError(f"ServerSocket: Received incorrect message length header {msg_len} bytes")

            # now retrieve rest of payload from buffer based on msg_len
            # TODO: handle rare edge case where only partial bytes were read before socket termination
            msg_body = b''
            received_data_size = 0
            while received_data_size < msg_len:
                data_size = msg_len - received_data_size
                received_data = self.client_socket.recv(data_size)
                received_data_size += len(received_data)
                msg_body += received_data

            # process received payload
            if msg_body == b'':
                raise RuntimeError("ServerSocket: Empty message for message length -{}".format(msg_len))
            elif msg_body:
                if received_data_size == msg_len:
                    msg = msg_body.decode("utf8", "ignore")
                else:
                    raise RuntimeError("ServerSocket: Received incomplete message.")

            # return reponse as JSON
            if format.upper() == "JSON":
                json_data = {}
                try:
                    json_data = json.loads(msg)
                except json.JSONDecodeError as e:
                    self.logger.error("ServerSocket: Bad JSON data - {},{}, {}".format(msg_len, msg, e))
                except Exception as e:
                    raise Exception("Bad formed message - {}{}, error- {}".format(msg_len, msg, e))

                return json_data
            else:
                return msg

        except socket.timeout as e:
            self.logger.error("ServerSocket: Timeout ({} seconds) from recv function".format((datetime.now() - time_started).seconds))
            # return status depending on received message size, if incomplete a Runtime Error should have been raised
            if len(msg_len_in_bytes) == self.MESSAGE_LENGTH and received_data_size == msg_len:
                return json.loads(msg)
            else:
                return json.loads("{}")
        except socket.error as e:
            self.logger.error(f"ServerSocket SocketError: {e}")
            raise socket.error(f"ServerSocket SocketError: {e}")
        except ValueError as e:
            raise ValueError(f"ServerSocket: ValueError - {e}")
        except RuntimeError as e:
            self.logger.error(f"ServerSocket: RuntimeError {e}")
            raise RuntimeError(f"ServerSocket: RuntimeError {e}")
        except Exception as e:
            raise Exception(f"ServerSocket: Exception {e}")

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
