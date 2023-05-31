#!/usr/bin/python

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

import dss
from minio.error import BucketAlreadyOwnedByYou


class DssClientLib(object):
    def __init__(self, s3_endpoint, access_key, secret_key, logger=None):
        self.s3_endpoint = "http://" + s3_endpoint
        self.logger = logger
        self.status = False
        self.dss_client = self.create_client(self.s3_endpoint, access_key, secret_key)

    def create_client(self, endpoint, access_key, secret_key):
        """
        Create dss_client
        :param endpoint:
        :param access_key:
        :param secret_key:
        :return:
        """
        dss_client = None
        try:
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

    def putObject(self, bucket=None, file=""):
        """
        A wrapper function of actual dss_client S3 upload function
        :param bucket:
        :param file:
        :return:
        """
        if file:
            object_key = file
            if file.startswith("/"):
                object_key = file[1:]
            ret = self.put_object(object_key, file)
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

    def getObject(self, bucket=None, object_key="", dest_file_path=""):
        """
        Download the objects from S3 storage and store in a local or share path.
        :param bucket: None # Not required
        :param object_key: required to get object
        :param dest_file_path: file path in which object should be copied.
        :return:
        """
        if object_key and dest_file_path:
            ret = self.get_object(object_key, dest_file_path)
            if ret == 0:
                return True
            if ret == 1:
                self.logger.error("Retry downloading object for key - {}".format(object_key))
                if self.get_object(object_key, dest_file_path) == 0:
                    return True
        return False

    def get_object(self, object_key, dest_file_path):
        """
        Download the objects from S3 storage and store in a local or share path.
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
            self.logger.execp("FileIOError - key:{}, {}".format(object_key, e))
            ret = 1
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(object_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - {} , {}".format(object_key, e))
        except dss.GenericError as e:
            self.logger.excep("GenericError - {} , {}".format(object_key, e))
            ret = 1
        except Exception as e:
            self.logger.excep("OtherException - {} , {}".format(object_key, e))
        return ret

    def listObjects(self, bucket=None, prefix="", delimiter="/"):
        """
        List object-keys under a specified prefix.
        The getObjects function has 3rd argument common_prefix should be True
        :param bucket: None ( for dss_client ) , For minio and boto3 there should be an bucket already created.
        :param prefix: A object key prefix
        :param delimiter: Default is "/" to receive first level object keys.
        :return: List of object keys.
        """
        try:
            objects = self.dss_client.getObjects(prefix, delimiter, True)
            while True:
                try:
                    for obj_key in objects:
                        yield obj_key
                except dss.NoIterator:
                    break
                except Exception as e:
                    self.logger.info("ListObjects {} - {}".format(prefix, e))
        except dss.FileIOError as e:
            self.logger.execp("FileIOError - key:{}, {}".format(obj_key, e))
        except dss.NoIterator as e:
            self.logger.excep("NoIterator - {}".format(e))
        except dss.NetworkError as e:
            self.logger.execp("NetworkError - key:{}, {}".format(obj_key, e))
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResouceError - {}".format(e))
        except dss.GenericError as e:
            self.logger.excep("GenericError {}- {}".format(prefix, e))
