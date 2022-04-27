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
import time
import hashlib
import socket
import queue
from multiprocessing import Process, Queue, Value


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
    :return:
    ret = { ret==0  indicate success
            ret !=0 failure
          }
    console_output = stdout gets returned
    """
    ret = 0
    console_output = ""
    std_out_default = sys.stdout
    #if not cmd.startswith('sudo'):
    #        cmd = 'sudo -u {} '.format(user_id) + cmd

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

def get_s3_prefix(path1, path2):
    """
    Validate prefix for minio S3 and return the same.
    :param logger
    :param path1: first part of prefix
    :param path2: second part of prefix
    :return: a valid s3 prefix
    """
    if path1.startswith("/"):
        path1 = path1[1:]
    if not path1.endswith("/"):
        path1 += "/"
    if path2.startswith("/"):
        path2 = path2[1:]

    return path1+path2



def validate_s3_prefix(prefix):
    """
    Validate a given prefix. A S3 prefix should start without "/" and end with "/".
    <prefix string>/
    :param logger: multiprocessing logger object
    :param prefix: a string
    :return: Success/Failure
    """
    if prefix.startswith("/") or not prefix.endswith("/"):
        print("ERROR: WRONG specification of prefix. Should be in the format of <prefix>/ => {}".format(prefix))
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


def create_file_path(path1, path2):
    """
    Create a complete file path
    :param path1: first part
    :param path2: second part
    :return:
    """
    if path1.endswith("/"):
        path1 = path1[1:]
    if not path2.startswith("/"):
        path2 = "/" + path2

    return path1 + path2




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
        index +=1

    return delimiter_index


def file_open(file_path, mode="r", logger=None):
    FH = None
    try:
        FH = open(file_path, mode)
    except OSError as e:
        if logger:
            logger.error(e)
        else:
            print("ERROR: {}".format(e))
    return FH

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
        self.mode = (kwargs.get("mode","r")).lower()
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
        if data and self.mode in ["w","a"]:
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
    
    

    


