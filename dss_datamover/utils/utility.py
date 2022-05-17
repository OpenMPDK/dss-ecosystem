#!/usr/bin/python

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

import os
import sys
import subprocess
import traceback
import ntplib
import time
import hashlib
import paramiko
import socket
import queue

"""
Contains list of utility functions...
"""

OPERATION_STATUS = [
    "READY",
    "RUNNING",
    "FINISHED"
]


def exception(func):
    """
    Implementation of nested function for decorator.
    :param func: <function ptr>
    :return: depends on function return type.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            print("EXCEPTION: {} : {}".format(e, traceback.format_exc()))
            return False

    return wrapper


def exec_cmd(cmd="", output=False, blocking=False, user_id="ansible", password="password"):
    """
    Execute the specified command
    :param cmd: <string> a executable command.
    :param output: Output
    :param blocking: Blocking or non-blocking call
    :param user_id:
    :param password:
    :return:
    ret = { ret==0  indicate success
            ret !=0 failure
          }
    console_output = stdout gets returned
    """
    ret = 0
    console_output = ""
    std_out_default = sys.stdout
    if not cmd.startswith('sudo'):
        cmd = 'sudo -u {} '.format(user_id) + cmd

    try:
        # print("INFO: Execution Cmd - {}".format(cmd))
        if blocking:
            if output:
                result = subprocess.check_output(cmd.split(), shell=False, stderr=subprocess.STDOUT,
                                                 universal_newlines=False)
                console_output = result
            else:
                with open(os.devnull, "wb") as fh:
                    ret = subprocess.call(cmd.split(), shell=False, stdout=fh, stderr=subprocess.STDOUT)
        else:
            subprocess.Popen(cmd.split())

    except subprocess.CalledProcessError as e:
        console_output = e.output
        ret = e.returncode
        # print(console_output)
        # print("Return code -", ret)
        # print(traceback.format_exc())

    finally:
        if not output:
            os.stdout = std_out_default

    return ret, console_output


@exception
def ntp_time(host):
    """
    return the ntp server time for unified timing
    :return: <int> ntp time
    """

    client = ntplib.NTPClient()
    response = client.request(host)
    ntp_time = int(response.tx_time)
    return ntp_time


@exception
def epoch(ts):
    """
    Get epoche from timestamp
    :param ts:
    :return:
    """
    time_format = '%Y-%m-%d %H:%M:%S'
    ts_epoch = int(time.mktime(time.strptime(ts, time_format)))
    return ts_epoch


@exception
def get_file_path(base_dir, file_name):
    """
    Return absolute file path.
    :param base_dir: <string> a file base directory path
    :param file_name: <string> file name
    :return: <string> Complete file path.
    """
    file_path = os.path.abspath(base_dir + "/" + file_name)

    return file_path


@exception
def remoteExecution(host, username, password="", cmd="", blocking=False):
    """
    Remote execution of a command to the specified host
    :param host:
    :param username:
    :param password:
    :param cmd:
    :param blocking:
    :return:
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    client.connect(hostname=host, username=username, password=password)
    stdin, stdout, stderr = client.exec_command(cmd)

    if blocking:
        status = stdout.channel.recv_exit_status()
        stdout_lines = stdout.readlines()
        stderr_lines = stderr.readlines()
        client.close()
        return stdout_lines, stderr_lines, status
    else:
        return client, stdin, stdout, stderr


def get_s3_prefix(logger, nfs_cluster, prefix=None):
    """
    Validate prefix for minio S3 and return the same.
    :param logger:
    :param nfs_cluster:
    :param prefix: s3 prefix
    :return: s3 compatible prefix
    """
    if prefix:
        if validate_s3_prefix(logger, prefix):
            prefix_fields = prefix.split("/")
            nfs_server_ip = prefix_fields[0]
            if nfs_server_ip not in nfs_cluster:
                nfs_first_dir = prefix_fields[0]
                for nfs_server_ip, nfs_shares in nfs_cluster.items():
                    for nfs_share in nfs_shares:
                        if nfs_first_dir is nfs_share.split("/")[0]:
                            yield nfs_server_ip + "/" + nfs_first_dir + "/"
            else:
                yield prefix
    else:
        for nfs_server_ip in nfs_cluster:
            yield nfs_server_ip + "/"


def validate_s3_prefix(logger, prefix):
    """
    Validate a given prefix. A S3 prefix should start without "/" and end with "/".
    <prefix string>/
    :param logger: multiprocessing logger object
    :param prefix: a string
    :return: Success/Failure
    """
    if prefix.startswith("/") or not prefix.endswith("/"):
        logger.error("WRONG specification of prefix. Should be in the format of <nfs_server_ip>/<prefix>/ ")
        return False
    return True


def progress_bar(prefix=""):
    """
    Display clockwise progress bar with specified prefix.
    :param prefix: A prefix string gets prepended ahead of progress bar.
    :return:
    """
    for i in range(1, 4):
        if i % 3 == 1:
            sys.stdout.write("\r{} |".format(prefix))
        elif i % 3 == 2:
            sys.stdout.write("\r{} /".format(prefix))
        elif i % 3 == 0:
            sys.stdout.write("\r{} --".format(prefix))
        time.sleep(0.1)


def get_hash_key(**kwargs):
    """
    Generate a 32 byte md5 hash key for a string or object or file content.
    :param kwargs: tuple of type and data
    type: Type of arguments are being passed ["string", "object", "file"]
    data: Pass string content, object or file
    :return: A 32 byte hash_key
    """
    type = kwargs.get("type", None)
    logger = kwargs["logger"]
    hash_key = None
    if type:
        if type == "file":
            file_path = kwargs["data"]
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as fh:
                    data = fh.read()
                    hash_key = hashlib.md5(data).hexdigest()
                    # logger.debug("HashKey - {}".format(hash_key))
            else:
                logger.error("File {} doesn't exist!".format(file_path))
        elif type == "object":
            object = kwargs["data"]
            hash_key = hashlib.md5(object).hexdigest()
        elif type == "str":
            data = kwargs["data"]
            hash_key = hashlib.md5(data).hexdigest()
        else:
            logger.error("Unknown Type {} for hashkey generation.\n Supported types are string/object/file".format(type))

    return hash_key


def get_ip_address(logger, hostname_or_ip_address=None, ip_address_family="IPV4"):
    """
    Generate IP address from hostname and internet address family specified.
    If IP address specified instead of hostname, it returns IP address.
    :param logger:
    :param hostname_or_ip_address:
    :param ip_address_family:
    :return:
    """
    ip_address = None
    if hostname_or_ip_address:
        try:
            if ip_address_family.upper() == "IPV4":
                ip_address = socket.gethostbyname(hostname_or_ip_address)
            elif ip_address_family.upper() == "IPV6":
                ip_address = socket.getaddrinfo(hostname_or_ip_address,
                                                None,
                                                family=socket.AF_INET6,
                                                proto=socket.IPPROTO_TCP)[-1][-1][0]
            else:
                logger.error("WRONG IP Address Family ...")
        except Exception as e:
            logger.error("Failed getting IP address for hostname/ip_address \
                         :{},Family:{} \n- {}".format(hostname_or_ip_address, ip_address_family, e))
    else:
        logger.error("Hostname or IP Address not specified!")

    return ip_address


@exception
def is_prefix_valid_for_nfs_share(logger, **kwargs):
    """
    Check if the prefix exist for a nfs share?
    :param logger:
    nfs_share: nfs_share
    prefix: A s3 prefix starts not with "/" and ends with "/".
    """
    nfs_share = kwargs["share"]
    prefix = kwargs["prefix"]
    ip_address = kwargs["ip_address"]
    nfs_mount_prefix = ip_address + nfs_share

    if prefix.startswith(nfs_mount_prefix) or nfs_mount_prefix.startswith(prefix):
        return True
    # logger.warn("Prefix:{}, is not part of nfs_share: {}".format(prefix, nfs_share))  # Delete
    return False


@exception
def is_queue_empty(mp_queue=None):
    queue_empty = True
    if mp_queue:
        if mp_queue.qsize() == 0:
            try:
                value = mp_queue.get(timeout=1)
                queue_empty = False
                mp_queue.put(value)
            except queue.Empty:
                pass
            except Exception as e:
                print("empty queue - {}".format(e))
        else:
            queue_empty = False
    else:
        print("Multi-processing Queue is not passed")
    return queue_empty


def decode(bytes=None):
    result = None
    if bytes:
        try:
            result = bytes.decode('utf8', 'ignore')
        except UnicodeDecodeError as e:
            print("Decoding error - {}".format(e))
    return result


def encode(bytes=None):
    result = None
    if bytes:
        try:
            result = bytes.encode('utf8', 'ignore')
        except UnicodeEncodeError as e:
            print("Encoding error - {}".format(e))
    return result


def first_delimiter_index(data_str, delimiter):
    """
    Return first delimiter position in a string
    :param data_str:
    :param delimiter:
    :return:
    """
    delimiter_index = -1
    index = 0
    data_str_len = len(data_str)
    while index < data_str_len:
        if data_str[index] == delimiter:
            delimiter_index = index
            break
        index += 1

    return delimiter_index


def file_open(file_path, mode="r", logger=None):
    fh = None
    try:
        fh = open(file_path, mode)
    except OSError as e:
        if logger:
            logger.error(e)
        else:
            print("ERROR: {}".format(e))
    return fh


def file_close(file_handle, logger=None):
    try:
        if file_handle:
            file_handle.close()
    except OSError as e:
        if logger:
            logger.error(e)
        else:
            print("ERROR: {}".format(e))


class File:

    def __init__(self, **kwargs):
        self.file = kwargs.get("path", None)
        self.mode = (kwargs.get("mode", "r")).lower()
        self.handler = None
        self.logger = kwargs.get("logger", None)
        self.size = 0
        self.flush = kwargs.get("flush", False)

    def __del__(self):
        self.close()

    def open(self):
        """
        Open a file with a specified mode
        :return:
        """
        try:
            self.handler = open(self.file, self.mode)
        except OSError as e:
            if self.logger:
                self.logger.error(e)
            else:
                print("ERROR: {}".format(e))

    def close(self):
        """
        Close a file
        :return:
        """
        if self.handler:
            try:
                self.handler.close()
            except OSError as e:
                if self.logger:
                    self.logger.error(e)
                else:
                    print("ERROR: {}".format(e))

    def size(self):
        """
        Return file size in bytes
        :return:
        """
        size = 0
        if self.mode == "r":
            if not self.stat:
                self.stat = os.stat(self.file)
            size = self.stat.st_size
        elif self.mode == "a":
            if not self.stat:
                self.stat = os.stat(self.file)
                self.size = self.stat.st_size
            size = self.size
        elif self.mode == "w":
            size = self.size

        return size

    def write(self, data=""):
        """
        Write to to a file in the specified mode.
        :param data: byte of chars to write into file.
        :return: None
        """
        if data and self.mode in ["w", "a"]:
            if type(data) is not str:
                data = str(data)
            self.size += self.handler.write(data)
        if self.flush:
            self.handler.flush()

    def read(self):
        """
        Return bytes of char read from file.
        :return:
        """
        pass

    def readlines(self):
        """
        Read all the lines from
        :return:
        """
