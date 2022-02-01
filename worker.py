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
from multiprocessing import Process, Value, Queue, current_process
#from minio_client import MinioClient
from s3_client import S3
#from logger import MultiprocessingLogger
#import prctl
import time
import glob
from utils.utility import validate_s3_prefix


class Worker(object):
    def __init__(self, **kwargs):


        # Worker details
        self.id = kwargs.get("id", None)
        self.status = Value('i',0)
        self.worker_pid = Value('i',0)

        self.storage_name = kwargs["storage_name"]
        self.storage_format = kwargs["storage_format"]

        self.s3_config = kwargs.get("s3_config", {})
        self.s3_client = None
        self.bucket = None
        self.queue = kwargs.get("queue", None)
        self.logger = kwargs.get("logger", None)
        self.data_dirs = kwargs["data_dirs"]
        self.categories = kwargs["categories"]
        self.finished = kwargs["worker_finished"]

        self.process =None


    def __del__(self):
        self.stop()
        # time.sleep(1)

    def create_s3_client(self):
        """
        Create actual s3_client based on s3 credential
        :return: s3_client connection.
        """
        credentials = self.s3_config["credentials"]
        try:
            s3_client_lib_name = self.s3_config["client_lib"].lower()
            #if s3_client_lib_name == "minio":
            #    self.s3_client = MinioClient(minio_url, minio_access_key, minio_secret_key, self.logger)
            if s3_client_lib_name == "dss_client":
                from dss_client import DssClientLib
                self.s3_client = DssClientLib(credentials=credentials,
                                              logger=self.logger)
            elif s3_client_lib_name == "boto3":
                self.s3_client = S3(storage_name=self.storage_name,
                                    credentials=credentials,
                                    logger=self.logger)
            else:
                self.logger.error(
                    "S3 Client-{} doesn't exist! Supported S3 clients [\"minio\",\"dss_client\", \"boto3\"] ".format(
                        s3_client_lib_name))
        except Exception as e:
            self.logger.excep("BAD s3_client {}".format(e))


    def get_s3_client(self):
        """
        Return s3_client 
        """
        return self.s3_client

    def start(self):
        """
        Start the worker process
        :return: None
        """
        try:
            self.process = Process(target=self.run, args=())
            self.process.name = "Worker-{}".format(self.id)
            self.process.start()
        except Exception as e:
            self.logger.excep("{}".format(e))
            return

    def stop(self):
        """
        Stop the worker process
        :return: None
        """
        if self.status.value:
            self.status.value = 0

        # Make sure process is stopped
        if self.process and self.process.is_alive():
            time.sleep(2)
            try:
                self.process.terminate()
            except Exception as e:
                self.logger.excep("Unable to terminate Worker-{} - {}".format(self.id, e))
        self.logger.info("Worker-{} stopped!".format(self.id))

    def get_status(self):
        """
        Return the worker status!
        :return:
        """
        return self.status.value



    def run(self):
        """
        Run the actual process
        :return:
        """
        name = "DNN_worker_" + str(self.id)
        #prctl.set_name(name)
        #prctl.set_proctitle(name)
        if self.storage_format == "s3":
            self.create_s3_client()
            if not self.s3_client:
                self.status.value = -1
                self.logger.error("S3 Client is not initialized. Exit worker-{}".format(self.id))
                return

        self.worker_pid.value = current_process().pid
        #self.logger.info("Worker-{}, PID-{} started ...".format(self.id, self.worker_pid.value))

        # Complete the defined task and exit
        try:

            result = self.list(self.s3_client)
            self.logger.info("Worker-{}, Listed Images:{}".format(self.id,len(result)))
            self.queue.put(result)
        except Exception as e:
            self.logger.excep("{}".format(e))

        with self.finished.get_lock():
            self.finished.value +=1

    def list(self, s3_client=None):
        """
        Perform file level listing or S3 listing.
        :param s3_client:
        :return:
        """


        images = []
        # Iterate over all specified the NFS paths/ all the prefixes.
        for category_path in self.data_dirs:
            self.logger.info("Worker-{}, Listing datadir/prefix - {}".format(self.id,category_path))
            #if self.storage_format == "s3" and not validate_s3_prefix(category_path):
            #    continue
            category = category_path.split("/")[-1]
            category_index = self.categories.index(category)

            if self.storage_format == "fs":
                #self.logger.info("Creating dataset for directory: {}/".format(category_path))
                # for root_path, dirs, files in os.walk(category_path, topdown=True):
                #    for file in files:
                #        file_path = root_path + "/" + file
                #        self.images.append((file_path, category_index))
                path_list = glob.glob(category_path + '/*', recursive=False)
                for image_path in path_list:
                    images.append((image_path, category_index))
            else:
                if self.s3_config:
                    bucket = self.s3_config["bucket"]
                prefix = "{}/".format(category_path)
                #self.logger.info("INFO: Creating dataset for the prefix: {} from bucket:{}".format(prefix,bucket))
                for object_keys in s3_client.listObjects(bucket=bucket, prefix=prefix):
                    for object_key in object_keys:
                        #self.logger.info("Object Key:{}".format(object_key))
                        images.append((object_key, category_index))
            #with self.categories_count.get_lock():
            #    self.categories_count.value -=1

        #self.logger.info("Worker-{}:{}".format(self.id,len(images)))
        return images

