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
from conftest import MockSocket
import time

def prepare_send_data(msg, header_len, wrong_size=False):
    if wrong_size:
        msg_len = (str(len(msg)+3)).zfill(header_len)
    else:
        msg_len = (str(len(msg))).zfill(header_len)
    ret = msg_len + msg
    return ret

@pytest.mark.usefixtures("get_pytest_configs", "get_config_dict", "get_mock_logger", "get_header_length")
class TestSocketCommunication():
    """ Unit tests for both ClientSocket and ServerSocket objects"""
    def test_client_socket_connect_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        
        ret = client_socket.connect('1.2.3.4', 1234)
        assert ret is None

    def test_client_socket_connect_invalid_ip(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        
        with pytest.raises(ConnectionError, match=r"Invalid IP Address given - .*"):
            client_socket.connect('*.@.^.!', 1234)

    def test_client_socket_send_json_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)

        ret = client_socket.send_json(r'{}')
        assert ret == True

        ret = client_socket.send_json(get_config_dict)
        assert ret == True

    def test_client_socket_send_json_empty_msg(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        
        ret = client_socket.send_json(r'')
        assert ret == False
    
    def test_client_socket_send_json_timeout(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)

        client_socket.socket.status = MockSocket.STATUS_SOCKETTIMEOUT
        with pytest.raises(socket.timeout):
            ret = client_socket.send_json({'k1':'v1'})
            assert ret == False
    
    def test_client_socket_send_json_bad_json(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)

        ret = client_socket.send_json('Not Json')
        assert ret == True
        assert re.match(r'ClientSocket: BAD MSG - .*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json_normal(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        json_str = json.dumps(get_config_dict)
        short_str = 'short msg'

        client_socket.connect('1.2.3.4', 1234)
        client_socket.socket.sendall(prepare_send_data(json_str, get_header_length))
        ret = client_socket.recv_json("JSON")
        assert ret == get_config_dict
        
        client_socket.socket.sendall(prepare_send_data(short_str, get_header_length))
        ret = client_socket.recv_json("STRING")
        assert ret == short_str
    
    def test_client_socket_recv_json_empty_msg(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        msg = ''

        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: Exception .*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json_wrong_header_size(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length+1)
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: .*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json_wrong_body_size(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length, True)
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: Exception .*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json_wrong_recv_size(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length)
        client_socket.socket.status = MockSocket.STATUS_MISALIGNEDBUFSIZE
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: Exception.*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json_socket_timeout(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length)
        client_socket.socket.sendall(msg)
        client_socket.socket.status = MockSocket.STATUS_SOCKETTIMEOUT
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: Timeout .*', get_mock_logger.get_last('error'))

    def test_client_socket_recv_json_bad_json_data(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        bad_json_str = 'abcdefghijk!@#$0987654321'

        msg = prepare_send_data(bad_json_str, get_header_length)
        client_socket.socket.sendall(msg)
        ret = client_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ClientSocket: Bad JSON data - .*', get_mock_logger.get_last('error'))

    def test_client_socket_close_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)

        client_socket.close()
        assert client_socket.socket.status == MockSocket.STATUS_CLOSED

    def test_client_socket_close_error(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        client_socket = ClientSocket(get_config_dict, get_mock_logger)
        client_socket.connect('1.2.3.4', 1234)
        client_socket.socket.status = MockSocket.STATUS_EXCEPTION

        client_socket.close()
        assert re.match(r'Close socket.*', get_mock_logger.get_last('excep'))

    def test_server_socket_bind_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)

        ret = server_socket.bind('1.2.3.4', 1234)
        assert ret is None
        assert server_socket.socket.status == MockSocket.STATUS_LISTENING
        assert server_socket.client_socket is None
        assert re.match(r'Client is listening for message on .*', get_mock_logger.get_last('info'))

    def test_server_socket_bind_invalid_ip(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        
        with pytest.raises(ConnectionError, match=r'Invalid IP Address - .*'):
            server_socket.bind('1.2.3.256', 1234)
        assert re.match(r'Wrong ip_address - .*', get_mock_logger.get_last('error'))
    
    def test_server_socket_accept_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        
        assert server_socket.client_socket is not None
        assert re.match(r'Connected to .*', get_mock_logger.get_last('info'))
    
    def test_server_socket_send_json_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)

        ret = server_socket.send_json({})
        assert ret == False

        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        ret = server_socket.send_json(r'{}')
        assert ret == True
        ret = server_socket.send_json(get_config_dict)
        assert ret == True

    def test_server_socket_send_json_empty_msg(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        
        ret = server_socket.send_json(r'')
        assert ret == False
    
    def test_server_socket_send_json_connection_error(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()

        server_socket.socket.status = MockSocket.STATUS_CONNECTIONERROR
        ret = server_socket.send_json({'k1':'v1'})
        assert ret == False
        assert re.match(r'ConnectionError-.*', get_mock_logger.get_last('error'))
    
    def test_server_socket_send_json_bad_json(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()

        ret = server_socket.send_json('This is Not Json', format="JSON")
        assert ret == True
        assert re.match(r'ServerSocket: BAD MSG - .*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json_normal(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        json_str = json.dumps(get_config_dict)
        short_str = 'short msg'

        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        server_socket.client_socket.sendall(prepare_send_data(json_str, get_header_length))
        ret = server_socket.recv_json("JSON")
        assert ret == get_config_dict
        server_socket.client_socket.sendall(prepare_send_data(short_str, get_header_length))
        ret = server_socket.recv_json("STRING")
        assert ret == short_str
    
    def test_server_socket_recv_json_empty_msg(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        msg = ''

        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: Exception .*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json_wrong_header_size(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length+1)
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket:.*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json_wrong_body_size(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length, wrong_size=True)
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: Exception .*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json_wrong_recv_size(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length)
        server_socket.client_socket.status = MockSocket.STATUS_MISALIGNEDBUFSIZE
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: Exception.*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json_socket_timeout(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        json_str = json.dumps(get_config_dict)

        msg = prepare_send_data(json_str, get_header_length)
        server_socket.client_socket.sendall(msg)
        server_socket.client_socket.status = MockSocket.STATUS_SOCKETTIMEOUT
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: Timeout .*', get_mock_logger.get_last('error'))

    def test_server_socket_recv_json_bad_json_data(self, mocker, get_config_dict, get_mock_logger, get_header_length):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()
        bad_json_str = 'abcdefghijk!@#$0987654321'

        msg = prepare_send_data(bad_json_str, get_header_length)
        server_socket.client_socket.sendall(msg)
        ret = server_socket.recv_json("JSON")
        assert ret == {}
        assert re.match(r'ServerSocket: Bad JSON data - .*', get_mock_logger.get_last('error'))

    def test_server_socket_close_normal(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()

        server_socket.close()
        assert server_socket.socket.status == MockSocket.STATUS_CLOSED
        assert server_socket.client_socket.status == MockSocket.STATUS_CLOSED

    def test_server_socket_close_error(self, mocker, get_config_dict, get_mock_logger):
        mocker.patch('socket.socket', MockSocket)
        server_socket = ServerSocket(get_config_dict, get_mock_logger)
        server_socket.bind('1.2.3.4', 1234)
        server_socket.accept()

        server_socket.close()
        assert re.match(r'Closing Socket.*', get_mock_logger.get_last('error'))
