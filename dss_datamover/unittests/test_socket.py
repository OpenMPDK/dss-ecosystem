#!/usr/bin/python3

"""
# The Clear BSD License
#
# Copyright (c) 2023 Samsung Electronics Co., Ltd.
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

import pytest
import socket
import os
import json
import re
from socket_communication import ServerSocket, ClientSocket
from conftest import MockSocket, Status
import time


def prepare_send_data(msg, header_len, wrong_size=False):
    if wrong_size:
        msg_len = (str(len(msg) + 3)).zfill(header_len)
    else:
        msg_len = (str(len(msg))).zfill(header_len)
    ret = msg_len + msg
    return ret


@pytest.mark.usefixtures("get_pytest_configs", "get_config_dict", "get_mock_logger", "get_header_length")
class TestSocketCommunication():
    """ Unit tests for both ClientSocket and ServerSocket objects"""
    def test_client_socket_connect(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        # positive case
        ret = client_socket.connect(get_pytest_configs["default_ip"], 1234)
        assert ret is None
        # Invalid IP
        with pytest.raises(ConnectionError, match=r"Invalid IP Address given - .*"):
            client_socket.connect('*.@.^.!', 1234)

    def test_client_socket_send_json(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect(get_pytest_configs["default_ip"], 1234)
        # positive case
        ret = client_socket.send_json(r'{}')
        assert ret
        ret = client_socket.send_json(get_config_dict)
        assert ret
        # empty message
        ret = client_socket.send_json(r'')
        assert not ret
        # socket timeout
        client_socket.socket.status = Status.SOCKETTIMEOUT
        with pytest.raises(socket.timeout):
            ret = client_socket.send_json({'k1': 'v1'})
            assert not ret
        # Bad Message
        client_socket.socket.status = Status.NORMAL
        ret = client_socket.send_json('Not Json')
        assert ret
        assert re.match(r'ClientSocket: BAD MSG - .*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json(self, mocker, get_config_dict, get_mock_logger, get_header_length, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        json_str = json.dumps(get_config_dict)
        short_str = 'short msg'
        client_socket.connect(get_pytest_configs["default_ip"], 1234)
        # positive
        client_socket.socket.sendall(prepare_send_data(json_str, get_header_length))
        ret = client_socket.recv_json("JSON")
        assert ret == get_config_dict
        client_socket.socket.sendall(prepare_send_data(short_str, get_header_length))
        ret = client_socket.recv_json("STRING")
        assert ret == short_str
        # empty message
        msg = ''
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: .*', get_mock_logger.get_last('error'))
        # wrong header size
        json_str = json.dumps(get_config_dict)
        msg = prepare_send_data(json_str, get_header_length + 1)
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: .*', get_mock_logger.get_last('error'))
        # wrong body size
        msg = prepare_send_data(json_str, get_header_length, True)
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: .*', get_mock_logger.get_last('error'))
        # wrong recv size
        msg = prepare_send_data(json_str, get_header_length)
        client_socket.socket.status = Status.MISALIGNEDBUFSIZE
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: .*', get_mock_logger.get_last('error'))
        # socket timeout
        client_socket.socket.status = Status.NORMAL
        msg = prepare_send_data(json_str, get_header_length)
        client_socket.socket.sendall(msg)
        client_socket.socket.status = Status.SOCKETTIMEOUT
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: .*', get_mock_logger.get_last('error'))
        # Bad JSON data
        client_socket.socket.status = Status.NORMAL
        bad_json_str = 'abcdefghijk!@#$0987654321'
        msg = prepare_send_data(bad_json_str, get_header_length)
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: Bad JSON data - .*', get_mock_logger.get_last('error'))

    def test_client_socket_close(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect(get_pytest_configs["default_ip"], 1234)
        # positive
        client_socket.close()
        assert client_socket.socket.status == Status.CLOSED
        # negative
        client_socket.socket.status = Status.EXCEPTION
        client_socket.close()
        assert re.match(r'Close socket.*', get_mock_logger.get_last('excep'))

    def test_server_socket_bind(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        # positive
        ret = server_socket.bind(get_pytest_configs["default_ip"], 1234)
        assert ret is None
        assert server_socket.socket.status == Status.LISTENING
        assert server_socket.client_socket is None
        assert re.match(r'Client is listening for message on .*', get_mock_logger.get_last('info'))
        # invalid IP
        with pytest.raises(ConnectionError, match=r'Invalid IP Address - .*'):
            server_socket.bind('1.2.3.256', 1234)
        assert re.match(r'Wrong ip_address - .*', get_mock_logger.get_last('error'))

    def test_server_socket_accept(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind(get_pytest_configs["default_ip"], 1234)
        # positive
        server_socket.accept()
        assert server_socket.client_socket is not None
        assert re.match(r'Connected to .*', get_mock_logger.get_last('info'))

    def test_server_socket_send_json(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        # send before bind
        ret = server_socket.send_json({})
        assert not ret
        # positive
        server_socket.bind(get_pytest_configs["default_ip"], 1234)
        server_socket.accept()
        ret = server_socket.send_json(r'{}')
        assert ret
        ret = server_socket.send_json(get_config_dict)
        assert ret
        # empty message
        ret = server_socket.send_json(r'')
        assert not ret
        # connection error
        server_socket.socket.status = Status.CONNECTIONERROR
        ret = server_socket.send_json({'k1': 'v1'})
        assert not ret
        assert re.match(r'ConnectionError-.*', get_mock_logger.get_last('error'))
        # bad json
        server_socket.socket.status = Status.NORMAL
        ret = server_socket.send_json('This is Not Json', format="JSON")
        assert ret
        assert re.match(r'ServerSocket: BAD MSG - .*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json(self, mocker, get_config_dict, get_mock_logger, get_header_length, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        # positive
        json_str = json.dumps(get_config_dict)
        short_str = 'short msg'
        server_socket.bind(get_pytest_configs["default_ip"], 1234)
        server_socket.accept()
        server_socket.client_socket.sendall(prepare_send_data(json_str, get_header_length))
        ret = server_socket.recv_json("JSON")
        assert ret == get_config_dict
        server_socket.client_socket.sendall(prepare_send_data(short_str, get_header_length))
        ret = server_socket.recv_json("STRING")
        assert ret == short_str
        # empty message
        msg = ''
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: .*', get_mock_logger.get_last('error'))
        # wrong header size
        msg = prepare_send_data(json_str, get_header_length + 1)
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: .*', get_mock_logger.get_last('error'))
        # wrong body size
        msg = prepare_send_data(json_str, get_header_length, wrong_size=True)
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: .*', get_mock_logger.get_last('error'))
        # wrong recv size
        msg = prepare_send_data(json_str, get_header_length)
        server_socket.client_socket.status = Status.MISALIGNEDBUFSIZE
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: .*', get_mock_logger.get_last('error'))
        # socket timeout
        msg = prepare_send_data(json_str, get_header_length)
        server_socket.client_socket.sendall(msg)
        server_socket.client_socket.status = Status.SOCKETTIMEOUT
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: .*', get_mock_logger.get_last('error'))
        # Bad JSON
        server_socket.client_socket.status = Status.NORMAL
        bad_json_str = 'abcdefghijk!@#$0987654321'
        msg = prepare_send_data(bad_json_str, get_header_length)
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: Bad JSON data - .*', get_mock_logger.get_last('error'))

    def test_server_socket_close(self, mocker, get_config_dict, get_mock_logger, get_pytest_configs):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind(get_pytest_configs["default_ip"], 1234)
        server_socket.accept()
        # positive
        server_socket.close()
        assert server_socket.socket.status == Status.CLOSED
        assert server_socket.client_socket.status == Status.CLOSED
        # close error
        server_socket.socket.status = Status.EXCEPTION
        server_socket.close()
        assert re.match(r'Closing Socket.*', get_mock_logger.get_last('error'))
