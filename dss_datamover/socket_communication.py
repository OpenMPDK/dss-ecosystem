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

import ipaddress
import json
import socket
import time
from datetime import datetime

# IP address family
IP_ADDRESS_FAMILY = {
    "IPv4": socket.AF_INET,
    "IPv6": socket.AF_INET6
}

DEFAULT_CONNECTION_RETRY_DELAY = 2  # wait time before retrying connection of socket
DEFAULT_MAX_CONNECTION_TIME_THRESHOLD = 300  # 5 Minutes, maximum wait time for socket connection.
DEFAULT_RESPONSE_HEADER_LENGTH = 10  # Message length 10 bytes
DEFAULT_RECV_TIMEOUT = 60  # Wait to receive data from socket for 60 seconds.
LOG_INTERVAL = 1  # prevent high frequecy log to cause deadlock


class ClientSocket(object):
    def __init__(self, config, logger=None):
        self.logger = logger
        self.config = config
        self.socket = None
        self.last_exception_log_time = datetime.min

    def connect(self, host=None, port=None):
        """
        Connect to a socket with the specified host and port.
        :param host:
        :param port:
        :return:
        """
        if not all([host, port]):
            raise ConnectionError("Host or port not specified")

        try:
            ip_info = ipaddress.ip_address(host)
            if isinstance(ip_info, ipaddress.IPv4Address):
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            elif isinstance(ip_info, ipaddress.IPv6Address):
                self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        except:
            self.logger.error("Wrong ip_address - {}, Supported {}".format(host, IP_ADDRESS_FAMILY))
            raise ConnectionError(f"Invalid IP Address given - {host}")

        # configures socket to send data as soon as it is available, regardless of packet size
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        is_connection_refused = True
        connection_time_start = datetime.now()
        connection_retry_delay = (
            self.config.get("socket_options", {}).get("connection_retry_delay", DEFAULT_CONNECTION_RETRY_DELAY)
        )
        time_to_sleep = connection_retry_delay
        max_connection_time_threshold = (
            self.config.get("socket_options", {}).get("max_connection_time_threshold", DEFAULT_MAX_CONNECTION_TIME_THRESHOLD)
        )

        while is_connection_refused:
            is_connection_refused = False
            try:
                self.socket.connect((host, int(port)))
            except ConnectionRefusedError as e:
                self.logger.warn(f"ConnectionRefusedError - retrying to connect {time_to_sleep} - {e}")
                is_connection_refused = True
            except ConnectionError as e:
                self.logger.excep(f"{host}:{port}-ConnectionError - {e}")
            except socket.error as e:
                self.logger.excep(f"{host}:{port}-GenericException - {e}")
            except Exception as e:
                self.logger.error(f"{host}:{port}->{e}")
            # Add a delay in increasing order of 2 sec.
            time.sleep(time_to_sleep)
            time_to_sleep += connection_retry_delay
            if (datetime.now() - connection_time_start).seconds > max_connection_time_threshold:
                raise socket.timeout("Socket connection timeout=300sec !")

    def send_json(self, message=None):
        """
        Send a JSON formatted data.
        :param message: JSON/DICT
        :return: Success/Failure = Tue/False
        """
        if message and self.socket:
            msg_body = json.dumps(message)
            msg_len = (str(len(msg_body))).zfill(
                self.config.get("socket_options", {}).get("response_header_length", DEFAULT_RESPONSE_HEADER_LENGTH)
            )
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
        :return: Return received data in json format.
        """
        self.socket.settimeout(self.config.get("socket_options", {}).get("recv_timeout", DEFAULT_RECV_TIMEOUT))
        msg = "{}"
        msg_len = 0
        time_started = datetime.now()
        response_header_length = (
            self.config.get("socket_options", {}).get("response_header_length", DEFAULT_RESPONSE_HEADER_LENGTH)
        )

        try:
            # first receive message length from payload
            msg_len_in_bytes = b''
            msg_len_in_bytes = self.socket.recv(response_header_length)
            if not msg_len_in_bytes:
                raise Exception('Empty data received from the socket')

            msg_len = int(msg_len_in_bytes)
            if len(msg_len_in_bytes) != response_header_length:
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
        except Exception as e:
            if (datetime.now() - self.last_exception_log_time).seconds > LOG_INTERVAL:
                self.logger.error(f"ClientSocket: Exception {e}")
                self.last_exception_log_time = datetime.now()

        # return status depending on received message size, if incomplete a Runtime Error should have been raised
        if len(msg_len_in_bytes) == response_header_length and received_data_size == msg_len:
            return json.loads(msg)
        return json.loads("{}")

    def close(self):
        """
        Close client socket
        :return:
        """
        try:
            # self.socket.shutdown(socket.SHUT_RDWR)
            self.socket.close()
        except Exception as e:
            self.logger.excep("Close socket {}".format(e))


class ServerSocket(object):
    def __init__(self, config, logger=None):
        self.socket = None
        self.logger = logger
        self.config = config
        self.client_socket = None
        self.last_exception_log_time = datetime.min

    def bind(self, host=None, port=None):
        """
        Bind server socket
        :param host:
        :param port:
        :return:
        """
        if not all([host, port]):
            self.logger.error("ERROR: Host not specified")
            raise ConnectionError("Host or port not specified")

        port = int(port)

        try:
            ip_info = ipaddress.ip_address(host)
            if isinstance(ip_info, ipaddress.IPv4Address):
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            elif isinstance(ip_info, ipaddress.IPv6Address):
                self.socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        except:
            self.logger.error("Wrong ip_address - {}, Supported {}".format(host, IP_ADDRESS_FAMILY))
            raise ConnectionError(f"Invalid IP Address - {host}")

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
        except Exception as e:
            self.logger.error("Not able to bind to host={}:{}, {}".format(host, port, e))
            raise ConnectionError(f"Socket initialization failed - Address {host}:{port} bind error - {e}!")

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

            msg_len = (str(len(msg_body))).zfill(
                self.config.get("socket_options", {}).get("response_header_length", DEFAULT_RESPONSE_HEADER_LENGTH)
            )
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
        :return: Return received data in json format.
        """
        self.client_socket.settimeout(self.config.get("socket_options", {}).get("recv_timeout", DEFAULT_RECV_TIMEOUT))
        msg_len = None
        msg = "{}"
        time_started = datetime.now()
        response_header_length = (
            self.config.get("socket_options", {}).get("response_header_length", DEFAULT_RESPONSE_HEADER_LENGTH)
        )

        try:
            # first receive message length from payload
            msg_len_in_bytes = b''
            msg_len_in_bytes = self.client_socket.recv(response_header_length)
            if not msg_len_in_bytes:
                raise Exception("Emtpy data received from the socket")

            msg_len = int(msg_len_in_bytes.decode('utf8'))
            if len(msg_len_in_bytes) != response_header_length:
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
        except Exception as e:
            if (datetime.now() - self.last_exception_log_time).seconds > LOG_INTERVAL:
                self.logger.error(f"ServerSocket: Exception {e}")
                self.last_exception_log_time = datetime.now()

        # return status depending on received message size, if incomplete a Runtime Error should have been raised
        if len(msg_len_in_bytes) == response_header_length and received_data_size == msg_len:
            return json.loads(msg)
        return json.loads("{}")

    def close(self):
        """
        Close server side socket.
        :return: None
        """
        try:
            # self.client_socket.shutdown(socket.SHUT_RDWR)
            self.client_socket.close()
            time.sleep(1)
            self.socket.close()
        except Exception as e:
            self.logger.error("Closing Socket - {}".format(e))

