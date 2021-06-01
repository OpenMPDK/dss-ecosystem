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

import os
import sys
import subprocess
import traceback
import ntplib
import time
import hashlib
import paramiko
from multiprocessing import Process, Queue, Value, Lock

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


def exec_cmd(cmd="", output=False, blocking=False):
    """
    Execute the specified command
    :param cmd: <string> a executable command.
    :param output: Output
    :param blocking: Blocking or non-blocking call
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
            cmd = 'sudo -u root ' + cmd

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
    :param cmd:
    :param blocking:
    :return:
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    client.connect(host, username=username, password=password)
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
    :param cluster_ip
    :param prefix: s3 prefix
    :return: s3 compatible prefix
    """
    if prefix:
        if validate_s3_prefix(logger, prefix):
            prefix_fields = prefix.split("/")
            nfs_server_ip = prefix_fields[0]
            if nfs_server_ip not in nfs_cluster:
                nfs_first_dir  = prefix_fields[0]
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
    :param type: Type of arguments are being passed ["string", "object", "file"]
    :param data: Pass string content, object or file
    :return: A 32 byte hash_key
    """
    type = kwargs.get("type", None)
    logger = kwargs["logger"]
    hash_key= None
    if type:
      if type == "file":
        file_path = kwargs["data"]
        if file_path and os.path.exists(file_path):
          with open(file_path, "rb") as fh:
            data = fh.read()
            hash_key = hashlib.md5(data).hexdigest()
            #logger.debug("HashKey - {}".format(hash_key))
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