#!/usr/bin/python
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
import dss
from datetime import datetime
from minio.error import BucketAlreadyOwnedByYou


class DssClientLib(object):
    def __init__(self, **kwargs):
        self.credentials = kwargs["credentials"]
        self.config = kwargs.get("config", {})
        self.logger = kwargs["logger"]
        self.s3_endpoint = self.credentials["endpoint"]
        self.access_key = self.credentials["access_key"]
        self.secret_key = self.credentials["secret_key"]
        self.status = False
        self.dss_client_options = None
        self.dss_client = self.create_client(self.s3_endpoint, self.access_key, self.secret_key)
        self.object_keys_per_page_count = self.config.get("object_keys_per_page_count", 1000)

    def set_dss_client_options(self):
        """
        Set some S3 options as required by S3.
        :return:
        """
        try:
            self.dss_client_options = dss.clientOption()
            self.dss_client_options.requestTimeoutMs = self.config.get("request_timeout_ms", 10000)  # 10 sec
            self.dss_client_options.maxConnections = self.config.get("max_connections", 25)
            self.dss_client_options.httpRequestTimeoutMs = self.config.get("http_request_timeout_ms", 0)
            self.dss_client_options.connectTimeoutMs = self.config.get("connect_timeout_ms", 1000)  # 1000
            self.dss_client_options.enableTcpKeepAlive = self.config.get("enable_tcp_keep_alive", True)
            self.dss_client_options.tcpKeepAliveIntervalMs = self.config.get("tcp_keep_alive_interval_ms", 30000)  # 10 sec
        except Exception as e:
            self.logger.excep(f"DSS_CLIENT_OPTIONS: {e}")

    def create_client(self, endpoint, access_key, secret_key):
        """
        Create dss_client
        :param endpoint:
        :param access_key:
        :param secret_key:
        :return:
        """
        dss_client = None
        self.set_dss_client_options()

        try:
            if self.dss_client_options:
                dss_client = dss.createClient(endpoint, access_key, secret_key, self.dss_client_options)
            else:
                dss_client = dss.createClient(endpoint, access_key, secret_key)
            if not dss_client:
                self.logger.error("Failed to create s3 client from - {}".format(endpoint))
            else:
                self.status = True
        except BucketAlreadyOwnedByYou as e:  # Do nothing
            self.logger.info("Bucket already Owned by you! ..")
            self.status = True
        except dss.DiscoverError as e:
            self.logger.excep("DiscoverError -  {}".format(e))
        except dss.NetworkError as e:
            self.logger.excep("NetworkError - {} , {}".format(endpoint, e))
        return dss_client

    def putObject(self, bucket=None, source_file="", object_key=""):
        """
        A wrapper function of actual dss_client S3 upload function
        :param bucket:
        :param file:
        :return:
        """
        if source_file:
            if object_key:
                if object_key.startswith("/"):
                    object_key = object_key[1:]
            else:
                if source_file.startswith("/"):
                    object_key = file[1:]
            ret = self.put_object(object_key, source_file)
            # Re-Try to upload file again
            if ret == 0:
                return True
            elif ret == 1:
                self.logger.info("Re-Uploading Object for Key-{}".format(object_key))
                if self.put_object(object_key, file) == 0:
                    return True

        return False

    def put_object(self, object_key, file=""):
        """
        Upload a object to S3
        On success return 0,
        On certain exception, perform a retry
        :param object_key: A string not starting with forward slash "/"
        :param file: A file with complete path
        :return: Success = 0, Failure-Retry = 1, Failure = -1 (No-Retry)
        """
        try:
            ret = self.dss_client.putObject(object_key, file)
            if ret == -1:
                self.logger.error("Upload Failed for  key - {}".format(object_key))
                ret = 1
        except dss.FileIOError as e:
            self.logger.execp("FileIOError - key:{}, {}".format(object_key, e))
            ret = 1
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
            ret = -1
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResourceError - key:{}, {}".format(object_key, e))
            ret = -1
        except dss.GenericError as e:
            self.logger.excep("GenericError - key:{}, {}".format(object_key, e))
            ret = 1

        return ret

    def deleteObject(self, bucket=None, object_key=""):
        """
        A wrapper function on top of S3 deleteObject function from dss_client_lib
        :param bucket: None , Not used a place holder.
        :param object_key: A object key, a string not starting with forward slash "/".
        :return: Success/Failure => True/False.
        """
        if object_key:
            if self.delete_object(object_key) == 0:
                return True
            elif self.delete_object(object_key) == 1:
                self.logger.info("Retrying to delete Object for Key-{}".format(object_key))
                if self.delete_object(object_key) == 0:
                    return True
        return False

    def delete_object(self, object_key):
        """
        Delete a object for an ObjectKey from S3 storage.
        :param object_key:  A object key, a string not starting with forward slash "/".
                            <dir1>/<dir2>/file1
        :return:
                0 => Removed successfully from S3 storage
                1 => Failed to remove for some reason such as IO failure
                -1 => Failed to remove due some Network/NoResourceFound error.
        """
        ret = -1
        try:
            if self.dss_client.deleteObject(object_key) == 0:
                ret = 0
            elif self.dss_client.deleteObject(object_key) == -1:
                self.logger.error("deleteObject filed for key - {}".format(object_key))
                ret = 1
        except dss.FileIOError as e:
            self.logger.execp("FileIOError - key:{}, {}".format(object_key, e))
            ret = 1
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - key:{}, {}".format(object_key, e))
        except dss.GenericError as e:
            self.logger.excep("GenericError - key:{}, {}".format(object_key, e))
            ret = 1
        return ret

    def getObject(self, **kwargs):
        """
        Download the objects from S3 storage and store in a local or share path.
        :param bucket: None # Not required
        :param key: required to get object
        :param dest_file_path: file path in which object should be copied.
        :return:
        """
        object_key = kwargs["key"]
        buffer = kwargs["memory"]
        buffer_length = 0
        if object_key:
            if type(buffer) == bytearray:
                buffer_length = self.get_object_buffer(object_key, buffer)
            else:
                buffer_length = self.get_object_numpy_buffer(object_key, buffer)
            # if not buffer:
            #    self.logger.error("Retry downloading object for key - {}".format(object_key))
            #    buffer_length = self.get_object_buffer(object_key, buffer)

        return buffer_length

    def getObjectToFile(self, **kwargs):
        """
        Download the objects from S3 storage and store in a local or share path.
        :param bucket: None # Not required
        :param key: required to get object
        :param dest_file_path: file path in which object should be copied.
        :return:
        """
        bucket = kwargs["bucket"]
        object_key = kwargs["key"]
        dest_file_path = kwargs["dest_file_path"]
        if dest_file_path:
            directory = os.path.dirname(dest_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
        if object_key and dest_file_path:
            ret = self.get_object(object_key, dest_file_path)
            if ret == 0:
                return True
            if ret == 1:
                self.logger.warn("Retry downloading object for key - {}".format(object_key))
                if self.get_object(object_key, dest_file_path) == 0:
                    return True
        return False

    def get_object(self, object_key, dest_file_path):
        """
        Download the objects from S3 storage and store in a local or shared file path.
        :param object_key:  A object key is unique in S3 storage and doesn't start with forward slash "/"
        :param dest_file_path: A physical file path where object to be copied.
        :return: Success = 0, Failure-Retry = 1, Failure = -1 (No-Retry)
        """
        ret = -1
        try:
            if self.dss_client.getObject(object_key, dest_file_path) == 0:
                ret = 0
            else:
                self.logger.error("Download Failed for Object-Key - {}".format(object_key))
        except dss.FileIOError as e:
            self.logger.error("FileIOError - key:{}, {}".format(object_key, e))
            ret = 1
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - {} , {}".format(object_key, e))
        except dss.GenericError as e:
            self.logger.error("GenericError - {} , {}".format(object_key, e))
            ret = 1
        except Exception as e:
            self.logger.excep("OtherException - {} , {}".format(object_key, e))
        return ret

    def get_object_buffer(self, object_key, buffer):
        """
        Download the objects from S3 storage and store into a bytearray buffer.
        :param object_key:  A object key is unique in S3 storage and doesn't start with forward slash "/"
        :param dest_file_path: A physical file path where object to be copied.
        :return: Success = 0, Failure-Retry = 1, Failure = -1 (No-Retry)
        """
        buffer_length = 0
        try:
            buffer_length = self.dss_client.getObjectBuffer(object_key, buffer)
        except dss.FileIOError as e:
            self.logger.error("FileIOError - key:{}, {}".format(object_key, e))
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - {} , {}".format(object_key, e))
        except dss.GenericError as e:
            self.logger.error("GenericError - {} , {}".format(object_key, e))
        except AttributeError as e:
            raise NotImplementedError(f"NotImplemented - {e}")
        except Exception as e:
            self.logger.excep("OtherException - {} , {}".format(object_key, e))
        return buffer_length

    def get_object_numpy_buffer(self, object_key, buffer):
        """
        Download the objects from S3 storage and store that into numpy object buffer.
        :param object_key:  A object key is unique in S3 storage and doesn't start with forward slash "/"
        :param buffer: A numpy memory
        :return: Success = 0, Failure-Retry = 1, Failure = -1 (No-Retry)
        """
        buffer_length = 0
        try:
            buffer_length = self.dss_client.getObjectNumpyBuffer(object_key, buffer)
        except dss.FileIOError as e:
            self.logger.error("FileIOError - key:{}, {}".format(object_key, e))
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - {} , {}".format(object_key, e))
        except dss.GenericError as e:
            self.logger.error("GenericError - {} , {}".format(object_key, e))
        except AttributeError as e:
            raise NotImplementedError(f"NotImplemented - {e}")
        except Exception as e:
            self.logger.excep("OtherException - {} , {}".format(object_key, e))
        return buffer_length

    def listObjects(self, bucket=None, prefix="", delimiter="/"):
        """
        List object-keys under a specified prefix.
        The getObjects function has 3rd argument common_prefix should be True
        The forth argument is used to specify number of object keys are to be returned in a single page. (10000)
        :param bucket: None ( for dss_client ) , For minio and boto3 there should be an bucket already created.
        :param prefix: A object key prefix
        :param delimiter: Default is "/" to receive first level object keys.
        :return: List of object keys.
        """
        try:
            objects = self.dss_client.getObjects(prefix, delimiter, True, self.object_keys_per_page_count)
            while True:
                try:
                    object_keys = []
                    for obj_key in objects:
                        object_keys.append(obj_key)
                    yield object_keys
                except dss.NoIterator:
                    break
                except Exception as e:
                    self.logger.info("ListObjects {} - {}".format(prefix, e))

        except dss.FileIOError as e:
            self.logger.execp("FileIOError - key:{}, {}".format(object_key, e))
        except dss.NoIterator as e:
            self.logger.excep("NoIterator - {}".format(e))
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - {}".format(e))
        except dss.GenericError as e:
            self.logger.excep("GenericError {}- {}".format(prefix, e))
