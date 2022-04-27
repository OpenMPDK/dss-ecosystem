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
from multiprocessing import Process, Value, Queue, current_process
from s3_client import S3
import prctl
import time
from utils.utility import validate_s3_prefix
from utils.utility import exec_cmd, create_file_path, get_s3_prefix

class Worker(object):
    def __init__(self, **kwargs):
        # Worker details
        self.id = kwargs.get("id", None)
        self.status = Value('i', 0)
        self.worker_pid = Value('i', 0)

        # Configuration
        self.operation = kwargs["operation"].lower()
        self.source_storage_type = kwargs.get("source_storage_type", None)
        self.destination_storage_type =  kwargs.get("destination_storage_type", None)
        self.list_queue = kwargs.get("list_queue", None)
        self.logger = kwargs.get("logger", None)
        self.finished = kwargs["worker_finished"]

        # Copy operation.
        self.file_paths = kwargs.get("file_paths", [])
        self.replication_factor = kwargs.get("replication_factor", 1)
        self.file_copy_count = kwargs.get("file_copy_count", 0)
        # S3 configuration
        self.source_s3_config = kwargs["source_s3_config"]
        self.source_data_dirs = kwargs.get("source_data_dirs", [])
        self.source_s3_client = None

        self.destination_s3_config = kwargs.get("destination_s3_config", {})
        self.destination_dirs = kwargs.get("destination_dirs", [])
        self.destination_s3_client = None
        self.process =None

    def __del__(self):
        self.stop()
        # time.sleep(1)

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

    def create_s3_client(self, source=False):
        """
        Create S3 client for both boto3 and dss_client.
        :return: s3_client
        """
        s3_client = None
        try:
            if source:
                client_library = self.source_s3_config["client_lib"]["name"].lower()
                credentials = self.source_s3_config["credentials"]
                dss_client_options = self.source_s3_config["client_lib"].get("dss_client", {})
                self.source_s3_bucket = self.source_s3_config["bucket"]
            else:
                client_library = self.destination_s3_config["client_lib"]["name"].lower()
                credentials = self.destination_s3_config["credentials"]
                dss_client_options = self.destination_s3_config["client_lib"].get("dss_client", {})
                self.destination_s3_bucket = self.destination_s3_config["bucket"]

            if client_library == "dss_client":
                from dss_client import DssClientLib
                s3_client = DssClientLib(credentials=credentials,
                                         config=dss_client_options,
                                         logger=self.logger)
            elif client_library == "boto3":
                from s3_client import S3
                s3_client = S3(credentials=credentials,
                               logger=self.logger)
            else:
                self.logger.fatal(f"DSS Doesn't support {client_library} library")
        except Exception as e:
            self.logger.fatal(f"Worker-{self.id}, {e}")

        return s3_client

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
        name = f"SDG_worker_{self.id}"

        # Set S3 client
        if self.source_storage_type == "s3":
            self.source_s3_client = self.create_s3_client(source=True)
            if self.source_s3_client is None:
                self.logger.error(f"Worker-{self.id} Exit!")
                sys.exit(f"Worker-{self.id} Exit!")
        if self.destination_storage_type == "s3":
            self.destination_s3_client = self.create_s3_client()
            if self.destination_storage_type is None:
                self.logger.error(f"Worker-{self.id} Exit!")
                sys.exit(f"Worker-{self.id} Exit!")

        prctl.set_name(name)
        prctl.set_proctitle(name)
        self.worker_pid.value = current_process().pid
        # Complete the defined task and exit
        try:
            if self.operation == "list":
                self.list(self.source_s3_client)
            else:
                self.copy()
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
        for data_dir in self.source_data_dirs:
            self.logger.info(f"Worker-{self.id}, listing {data_dir}")
            if self.source_storage_type == "fs":
                for root_path, dirs, files in os.walk(data_dir, topdown=True):
                    for file in files:
                        file_path = root_path + "/" + file
                        images.append(file_path)
            elif self.source_storage_type == "s3":

                if self.source_s3_config:
                    bucket = self.source_s3_config["bucket"]
                if not data_dir.endswith("/"):
                    data_dir = data_dir + "/"
                self.logger.info(f"Prefix:{data_dir}")
                for object_keys in s3_client.listObjects(bucket=bucket, prefix=data_dir):
                    for object_key in object_keys:
                        images.append(object_key)
        self.logger.info("Worker-{}, Listed Images:{}".format(self.id, len(images)))
        self.list_queue.put(images)


    def copy(self):
        """
        Copy the files to the respective path with matching replication factor.
        :return:
        """
        self.logger.info(f"Worker-{self.id} started copy operation!")
        # Copy for each destination dir/prefix
        for destination_path in self.destination_dirs:
            for source_file in self.file_paths:
                # Skip for directory
                if os.path.isdir(source_file):
                    continue
                destination_file = source_file
                for index in range(self.replication_factor):
                    if self.replication_factor > 1:
                        # update file name with replication factor index
                        file_name = os.path.basename(source_file)
                        fields = file_name.split(".")
                        if len(fields) > 1:
                            file_name_temp = ".".join(fields[0:-1])
                            file_extension = fields[-1]
                            file_name = f"{file_name_temp}_{index}.{file_extension}"
                        else:
                            file_name = f"{file_name}_{index}" # File without any extension

                        file_path = os.path.dirname(source_file)
                        destination_file =  file_path + "/" +  file_name

                    # Perform file copy operation.
                    try:
                        if self.source_storage_type == "fs" and self.destination_storage_type == "fs":
                            destination_file_path = create_file_path(destination_path, destination_file)
                            self.file_copy(source_file, destination_file_path)
                        else:
                            self.object_copy(source=source_file,
                                             destination_basepath=destination_path,
                                             destination_file=destination_file)
                    except Exception as e:
                        self.logger.error(f"FileCopy:{e}")

                    with self.file_copy_count.get_lock():
                        self.file_copy_count.value +=1

    def object_copy(self, **kwargs):
        """
        Copy each file/object from file system/object storage to object storage/file systems
        :param kwargs:
        :return:
        """
        source = kwargs["source"]
        destination_basepath = kwargs["destination_basepath"]
        destination_file = kwargs["destination_file"]
        # Upload to DSS S3
        if self.source_storage_type == "fs" and self.destination_storage_type == "s3":
            destination = get_s3_prefix(destination_basepath, destination_file)
            self.logger.debug(f"Object Copy:{source}, Destination:{destination}")
            self.destination_s3_client.putObject(self.destination_s3_bucket, source,
                                                 destination)
        elif self.source_storage_type == "s3" and self.destination_storage_type == "fs":
            # Download from DSS S3 to FS
            destination_file_path = create_file_path(destination_basepath, destination_file)
            self.source_s3_client.getObjectToFile(bucket=self.source_s3_bucket,
                                                  key=source,
                                                  dest_file_path=destination_file_path)
        elif self.source_storage_type == "s3" and self.destination_storage_type == "s3":
            # Move data S3 to S3
            raise NotImplementedError("This feature is not supported")



    def file_copy(self, source_file, destination_file_path):
        """
        Copy file from a source path to a destination path.
        :return: None
        """
        try:
            dest_dir = os.path.dirname(destination_file_path)
            # Create directory if not exist
            if not os.path.exists(dest_dir):
                cmd_mkdir = f"mkdir -p {dest_dir}"
                mkdir_ret, mkdir_console = exec_cmd(cmd_mkdir, True, True)
                if mkdir_ret != 0:
                    self.logger.error(f"FileCopy: Failed to create directory {mkdir_console}")
                    return
            # File System copy
            cmd = f"cp {source_file} {destination_file_path} "
            copy_ret, copy_console = exec_cmd(cmd, True, True)
            if copy_ret != 0:
                self.logger.error(f"Copy Failed: Source-{source_file},"
                                  f"Destination-{destination_file_path}"
                                  f"\n{copy_console}")
        except Exception as e:
            self.logger.error(f"FileCopy: {e}")




