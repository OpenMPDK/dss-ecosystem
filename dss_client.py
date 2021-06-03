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
import os,sys
import dss
from datetime import datetime




class DssClientLib:
    def __init__(self, s3_endpoint, access_key, secret_key, logger=None):
        self.s3_endpoint = "http://" + s3_endpoint
        self.logger = logger
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
            dss_client =  dss.createClient(endpoint, access_key,secret_key)
            if not dss_client:
                self.logger.error("Failed to create s3 client from - {}".format(endpoint))
        except dss.DiscoverError as e:
            self.logger.excep("DiscoverError -  {}".format(e))
        except dss.NetworkError as e:
            #print("EXCEPTION: NetworkError - {}".format(e))
            self.logger.excep("NetworkError - {} , {}".format(endpoint, e))

        return dss_client

    def putObject(self, bucket=None,file=""):

        if file :
            object_key = file
            if file.startswith("/"):
                object_key = file[1:]
            try:
                ret = self.dss_client.putObject(object_key, file)
                if ret == 0:
                    return True
                elif ret == -1:
                    self.logger.error("Upload Failed for  key - {}".format(object_key))
            except dss.NoSuchResouceError as e:
                self.logger.excep("NoSuchResourceError putObject - {}".format(e))
            except dss.GenericError as e:
                self.logger.excep("putObject {}".format(e))
        return False

    def listObjects_old(self, bucket=None,  prefix="", delimiter="/"):
        object_keys = []
        try:
            object_keys = self.dss_client.listObjects(prefix, delimiter)
            #if object_keys:
            #    yield object_keys
        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResourceError - {}".format(e))
        except dss.GenericError as e:
            self.logger.excep("listObjects - {}".format(e))

        return object_keys

    def deleteObject(self, bucket=None, object_key=""):
        #self.logger.delete("......{}".format(object_key))
        if object_key:
            try:
                if self.dss_client.deleteObject(object_key) == 0:
                    return True
                elif self.dss_client.deleteObject(object_key) == -1:
                    self.logger.error("deleteObject filed for key - {}".format(object_key))
            except dss.NoSuchResouceError as e:
                self.logger.excep("deleteObject - {}, {}".format(object_key,e))
            except dss.GenericError as e:
                self.logger.excep("deleteObject - {}, {}".format(object_key, e))
        return False


    def getObject(self, bucket=None, object_key="", dest_file_path=""):
        """
        Download the objects from S3 storage and store in a local or share path.
        :param bucket: None # Not required
        :param object_key: required to get object
        :param dest_file_path: file path in which object should be copied.
        :return:
        """
        if object_key and dest_file_path:
            try:
                if self.dss_client.getObject(object_key, dest_file_path) == 0:
                    return True
            except dss.NoSuchResouceError as e:
                self.logger.excep("getObject - {} , {}".format(object_key, e))
            except dss.GenericError as e:
                self.logger.excep("getObject - {} , {}".format(object_key, e))
            except Exception as e:
                self.logger.excep("ObjectKey-{} , {}".format(object_key, e))
        return False

    def listObjects(self, bucket=None,  prefix="", delimiter="/"):
        """
        List object keys under a specified prefix .
        :param bucket: None ( for dss_client ) , For minio and boto3 there should be an bucket already created.
        :param prefix: A object key prefix
        :param delimiter: Default is "/" to receive first level object keys.
        :return: List of object keys.
        """
        object_keys = []

        try:
            iterator = self.dss_client.getObjects(prefix, delimiter)
            while True:
                try:
                    for obj_key in iterator:
                        object_keys.append(obj_key)
                except dss.NoIterator:
                    break
                except Exception as e:
                    self.logger.info("ListObjects {} - {}".format(prefix, e))

        except dss.NoSuchResouceError as e:
            self.logger.excep("NoSuchResourceError - {}".format(e))
        except dss.GenericError as e:
            self.logger.excep("listObjects {}- {}".format(prefix, e))

        return object_keys
