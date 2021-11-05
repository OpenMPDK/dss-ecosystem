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

import os, sys
from utils.utility import exception, exec_cmd, get_hash_key
from multiprocessing import Value, Manager, current_process
from minio_client import MinioClient
from s3_client import S3
import json
from datetime import datetime
import time

task_id = Value('i', 1)

mgr = Manager()
ds = mgr.dict()

"""
Need to be updated 
"""


@exception
def put(s3_client, **kwargs):
    """
    Upload operation
    :param s3_client:
    :param kwargs: contains params, status_queue, logger
    :return:
    """
    params = kwargs["params"]
    status_queue = kwargs["status_queue"]
    logger = kwargs["logger"]
    operation_progress_status_counter = kwargs["operation_progress_status_counter"]
    index_data = params.get("data", {})
    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")

    # Data Integrity
    data_integrity = kwargs.get("data_integrity", False)
    dryrun = params.get("dryrun", False)
    failed_files = []

    success = 0
    failure_files_size = 0
    if s3_client:
        # start_time = datetime.now()
        for file_name in index_data["files"]:
            file = os.path.abspath(index_data["dir"] + "/" + file_name)
            if os.access(file, os.R_OK):
                operation_progress_status_counter.value += 1
                try:
                    if dryrun:
                        # Read file for the purpose of testing
                        #with open(file, "rb") as FH:
                        #    lines = FH.readlines()
                        #lines = []
                        success += 1
                    else:
                        if s3_client.putObject(minio_bucket, file):
                            success += 1
                        else:
                            failure_files_size += os.path.getsize(file)
                            failed_files.append(file_name)
                        operation_progress_status_counter.value += 1
                except Exception as e:
                    logger.excep("PUT - {}".format(e))
                    failure_files_size += os.path.getsize(file)
                    failed_files.append(file_name)
            else:
                if not os.path.exists(file):
                    logger.error("{},PID-{} File-{} doesn't exist".format(current_process().name,
                                                                          current_process().pid, file))
                else:
                    logger.error("{},PID-{} Read access denied -{}".format(current_process().name,
                                                                           current_process().pid, file))
                failed_files.append(file_name)
    else:
        logger.error("Unable to connect to S3 Storage for upload")

    if not data_integrity:
        # Update following section for upload status.
        status_message = {"success": success, "failure": (len(index_data["files"]) - success) ,
                          "dir": index_data["dir"] ,
                          "failed_files": failed_files,
                          "size" : failure_files_size}
        status_queue.put(status_message)

@exception
def list(s3_client, **kwargs):
    """
    List the Object keys from lower level directory. If not then, create a task, so that can be processed by other
    worker.
    :param params:
    :param task_queue: Holds the task
    :param index_data_queue:  Holds the list index message to be distributed among the client nodes.
    :param logger: A logger with shared queue used among the running processes
    :param listing_progress: it holds the value of outstanding prefix
    :return:
    """
    worker_id = kwargs["worker_id"]
    params = kwargs["params"]
    logger = kwargs["logger"]
    task_queue = kwargs["task_queue"]  # Contains task Object
    index_data_queue = kwargs["index_data_queue"]  # Contains index_data
    index_data_count = kwargs["index_data_count"]
    index_msg_count = kwargs["index_msg_count"]
    listing_progress = kwargs["listing_progress"]  # The shared memory is used to hold progress status.
    listing_status = kwargs["listing_status"]
    listing_only = kwargs["listing_only"]
    listing_objectkey_queue = kwargs["listing_objectkey_queue"]
    listing_based_on_indexing = params["listing_based_on_indexing"]
    dump_object_keys_path = params["dest_path"] # Dump object keys to the file on this specified path. 
    max_index_size = params["max_index_size"]

    prefix = None
    if params.get("data", {}) and params["data"].get("prefix", None):
        prefix = (params["data"]["prefix"]).strip()

    # That mean lowest level of directory.
    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")
    s3_client_library = s3config.get("client_lib", "minio_client")

    object_keys_iterator = s3_client.listObjects(minio_bucket, prefix)
    if object_keys_iterator:
        object_keys_count = 0
        for result in list_object_keys(object_keys_iterator, max_index_size, s3_client_library):
            if "object_keys" in result:
                index_data_message = {"dir": prefix, "files": result["object_keys"]}
                object_keys_count += len(result["object_keys"])
                if listing_only.value and dump_object_keys_path:
                    result.update({"prefix": prefix})
                    listing_objectkey_queue.put(result)
                else:
                    index_data_queue.put(index_data_message)
                    with index_msg_count.get_lock():
                        index_msg_count.value += 1
            else:
                if not listing_based_on_indexing:
                    with listing_progress.get_lock():
                        listing_progress.value += 1
                    task = Task(operation="list", data=result, s3config=params["s3config"],
                            max_index_size=max_index_size, listing_based_on_indexing=listing_based_on_indexing,
                            dest_path=dump_object_keys_path)
                    task_queue.put(task)

        with index_data_count.get_lock():
            index_data_count.value += object_keys_count
    else:
        logger.error("No object keys belongs to the prefix-{}".format(prefix))

    if listing_status.value != 1:
        listing_status.value = 1

    with listing_progress.get_lock():
        if listing_progress.value:
            listing_progress.value -= 1

def distributed_list(s3_client, **kwargs):
    """
    Perform S3 LISTing from multiple nodes. Distribute the prefix_dir among the clients
    :param s3_client:
    :param kwargs:
    :return:
    """
    params = kwargs["params"]
    logger = kwargs["logger"]
    status_queue = kwargs["status_queue"]

    prefix = None
    if params.get("data", {}) and params["data"].get("dir", None):
        prefix = (params["data"]["dir"]).strip()

    # That mean lowest level of directory.
    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")
    s3_client_library = s3config.get("client_lib", "minio_client")

    object_keys_iterator = s3_client.listObjects(minio_bucket, prefix)
    object_keys_count =0
    max_index_size = sys.maxsize
    for result in list_object_keys(object_keys_iterator, max_index_size, s3_client_library):
        if "object_keys" in result:
            object_keys_count += len(result["object_keys"])
            result.update({"prefix": prefix})
            status_queue.put(result)
    if object_keys_count == 0:
        result = {"prefix": prefix, "object_keys": []}
        status_queue.put(result)

def list_object_keys(object_keys_iterator, max_index_size, s3_client_lib):
    """
    Iterate over the object keys and generate a message which holds a prefix and object keys underneath of the prefix.
    A prefix dir is ended with a forward slash , ex <a>/<b>/<c>/
    A object key doesn't have any ending forward slash, ex <a>/<b>/<c>/object1
    :param object_keys_iterator: Returned object keys iterator,
    :param max_index_size:
    :return:
    """

    object_keys = []
    for obj_key_iter in object_keys_iterator:
        if s3_client_lib.lower() == "minio":
            obj_key = obj_key_iter.object_name
        elif s3_client_lib.lower() == "dss_client":
            obj_key = obj_key_iter
        else:
            obj_key = ""
        # To handle a scenario in which directory has both file and directory
        if obj_key.endswith("/"):
            yield {"prefix": obj_key}
        else:
            obj_key = obj_key.split("/")[-1]
            if len(object_keys) == max_index_size:
                yield {"object_keys": object_keys}
                object_keys = [obj_key]
            else:
                object_keys.append(obj_key)
    if object_keys:
        yield {"object_keys": object_keys}


@exception
def get(s3_client, **kwargs):
    params = kwargs["params"]
    status_queue = kwargs["status_queue"]
    logger = kwargs["logger"]
    dest_path = params.get("dest_path", "")
    user_id = params["user_id"]

    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")
    dryrun = params.get("dryrun", False)

    # Data Integrity
    data_integrity = kwargs.get("data_integrity", False)
    file_hash_map = kwargs.get("file_hash_map", {})
    success = 0
    object_keys = params["data"]
    failed_files = []

    if s3_client:
        # Create directory if not exist
        ret = 0
        dest_dir = dest_path + "/" + object_keys["dir"]
        try:
            if not os.path.exists(dest_dir):
                command = "mkdir -p {}".format(dest_dir)
                ret, console = exec_cmd(command, True, True, user_id)
                if ret:
                    logger.error("Local Prefix directory {} creation FAILED\n\t{}".format(dest_dir, console))
                    if 'No space left' in console:
                        logger.fatal('No space left on the file system')
        except Exception as e:
            logger.excep("Prefix-{}\n{}".format(object_keys["dir"], e))

        if ret == 0:
            for object_key in object_keys["files"]:
                if dryrun:  # Dry run
                    success += 1
                else:  # Actual operation
                    dest_file_path = dest_dir + "/" + object_key
                    object_key = object_keys["dir"] + object_key
                    # logger.debug("Obj_Key:{}, Dest_Path:{}".format(object_key,dest_file_path))
                    try:
                        if s3_client.getObject(minio_bucket, object_key, dest_file_path):
                            if data_integrity:
                                hash_key = get_hash_key(type='file', data=dest_file_path, logger=logger)
                                file_name = object_key.split("/")[-1]
                                if file_hash_map[file_name] == hash_key:
                                    success += 1
                                else:
                                    logger.error("Failed DataIntegrity for Object Key - {}".format(object_key))
                            else:
                                success += 1
                        else:
                            logger.error("Failed to download object - {}".format(object_key))
                            failed_files.append(object_key)
                    except Exception as e:
                        if 'No space left on device' in str(e):
                            break
    else:
        logger.error("Unable to connect to S3 ObjectStorage for download")

    # Update following section for upload status.
    prefix_dir = object_keys["dir"]
    if data_integrity:
        prefix_dir = "/" + prefix_dir[:-1]
    status_message = {"success": success, "failure": (len(object_keys["files"]) - success),
                      "failed_files": failed_files, "dir": prefix_dir}
    status_queue.put(status_message)


@exception
def delete(s3_client, **kwargs):
    params = kwargs["params"]
    status_queue = kwargs["status_queue"]
    logger = kwargs["logger"]

    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")

    success = 0
    object_keys = params["data"]
    prefix = object_keys["dir"]
    if s3_client:
        for object_key in object_keys["files"]:
            object_key = prefix + object_key
            # logger.debug("TASK: Going to removed object key {}".format(object_key))
            if params.get("dryrun", False):
                # Dry run
                success += 1
            else:
                # Actual operation
                if s3_client.deleteObject(minio_bucket, object_key):
                    # logger.debug("TASK: Removed object key {}".format(object_key))
                    success += 1
    else:
        logger.error("Unable to connect to Minio for upload with {}".format(s3config))

    # Update following section for upload status.
    status_message = {"success": success, "failure": (len(object_keys["files"]) - success), "dir": prefix}
    status_queue.put(status_message)


@exception
def data_integrity(s3_client, **kwargs):
    """
    - For each prefix for leaf node perform upload operation
      - Before upload for each file , get the md5sum of that and store it in a buffer as <file> => <md5sum>
    - Download all files for the same prefix and perform md5sum for each file
      - Compare the same with previously stored hash key.
      - Send the status to master.
    :param s3_client:
    :param kwargs:
    :return:
    """
    logger = kwargs["logger"]
    skip_upload = kwargs["skip_upload"]
    # Upload first
    data_integrity = True
    file_hash_map = {}
    params = kwargs.get("params", {})
    data = params["data"]
    user_id = params["user_id"]
    status_queue = kwargs["status_queue"]
    logger.info("Checking DataIntegrity for the prefix - {}".format(data["dir"]))
    if not skip_upload:
        logger.info("** DataIntegrity: Performing PUT operation - Prefix-{} **".format(data["dir"]))

    try:
        for file_name in data["files"]:
            file = os.path.abspath(data["dir"] + "/" + file_name)
            if os.access(file, os.R_OK):
                file_content_hash_key = get_hash_key(type="file", data=file, logger=logger)
                file_hash_map[file_name] = file_content_hash_key
                if not skip_upload:
                    # Create datastructure reuired for upload
                    upload_data = {"dir": data["dir"], "files": [file_name]}
                    params["data"] = upload_data
                    put(s3_client, params=params, data_integrity=data_integrity, status_queue=status_queue,
                        logger=logger)
            else:
                if not os.path.exists(file):
                    logger.error("File-{} doesn't exist".format(file))
                else:
                    logger.error("Read access denied - {}".format(file))
    except Exception as e:
        logger.excep("DataIntegrity-PUT:{}".format(e))
    ## Download
    # - Fine tune the parameters for GET
    # Update data with object keys as expected by GET function
    # Create prefix_dir from directory
    data["dir"] = data["dir"][1:] + "/"
    params["data"] = data
    logger.info("** DataIntegrity: Performing GET operation - Prefix-{} **".format(data["dir"]))
    get(s3_client, params=params,
        status_queue=status_queue,
        data_integrity=data_integrity,
        file_hash_map=file_hash_map,
        logger=logger)

    # Remove the local files
    try:
        download_path = os.path.abspath(params["dest_path"] + "/" + data["dir"])
        logger.info("DataIntegrity: Removing files under the download path - {}".format(download_path))
        for file_name in data["files"]:
            downloaded_file_path = os.path.abspath(download_path + "/" + file_name)
            if os.path.exists(downloaded_file_path):
                command = "rm -rf {}".format(downloaded_file_path)
                ret, console = exec_cmd(command, True, True, user_id)
                if ret:
                    logger.error("Failed to remove file -{}\n {}".format(downloaded_file_path, console))
            else:
                logger.error("File \"{}\" doesn't exist ".format(downloaded_file_path))
    except Exception as e:
        logger.excep("DataIntegrity-Remove {}".format(e))


class Task(object):
    def __init__(self, **kwargs):
        self.id = task_id.value
        with task_id.get_lock():
            task_id.value += 1
        self.operation = kwargs["operation"].lower()
        kwargs.pop("operation")
        self.params = kwargs

    def start(self, **queue):
        self.logger = queue["logger"]
        self.task_queue = queue["task_queue"]
        self.index_data_queue = queue["index_data_queue"]

        s3_client = queue["s3_client"]
        # self.logger.debug("Task: Operation-{}, Data-{}".format(self.operation, self.params["data"]))
        try:
            if self.operation == "put":
                put(s3_client, params=self.params,
                    status_queue=queue["status_queue"],
                    operation_progress_status_counter=queue["operation_progress_status_counter"],
                    logger=self.logger)
            elif self.operation == "list":
                if self.params.get("distributed", False):
                    distributed_list(s3_client, params=self.params, status_queue=queue["status_queue"],
                                     logger=self.logger)
                else:
                    list(s3_client, params=self.params,
                         worker_id=queue["worker_id"],
                         task_queue=queue["task_queue"],
                         index_data_queue=queue["index_data_queue"],
                         index_data_count=queue["index_data_count"],
                         index_msg_count=queue["index_msg_count"],
                         logger=self.logger,
                         listing_progress=queue["listing_progress"],
                         listing_status=queue["listing_status"],
                         listing_only=queue["listing_only"],
                         listing_objectkey_queue=queue["listing_objectkey_queue"])
            elif self.operation == "del":
                delete(s3_client, params=self.params,
                       status_queue=queue["status_queue"],
                       logger=self.logger)

            elif self.operation == "get":
                get(s3_client, params=self.params,
                    status_queue=queue["status_queue"],
                    logger=self.logger)
            elif self.operation == "test":
                data_integrity(s3_client, params=self.params,
                               status_queue=queue["status_queue"],
                               skip_upload=queue["skip_upload"],
                               logger=self.logger)
            elif self.operation == "indexing":
                indexing(data=self.params["data"],
                         nfs_cluster=self.params["nfs_cluster"],
                         nfs_share=self.params["nfs_share"],
                         task_queue=queue["task_queue"],
                         index_data_queue=queue["index_data_queue"],
                         logger=self.logger,
                         progress_of_indexing=queue["progress_of_indexing"],
                         progress_of_indexing_lock=queue["progress_of_indexing_lock"],
                         index_data_count=queue["index_data_count"],
                         index_msg_count=queue["index_msg_count"],
                         max_index_size=self.params.get("max_index_size", 10),
                         indexing_started_flag=queue["indexing_started_flag"],
                         index_data_queue_size=queue["index_data_queue_size"],
                         prefix_index_data=queue["prefix_index_data"],
                         standalone=queue["standalone"],
                         params=self.params
                         )

        except Exception as e:
            self.logger.excep("{}:Task: {}!".format(__file__, e))

    def stop(self):
        pass


def indexing_dir(**kwargs):
    """
    Create a metadata index during PUT operation only.
    First Phase:
    - Create a JOSN structure for each NFS share with a NFS share name.
      Required information - directory name, file name, counts, total size of directory.

    :return:
    """
    dir = kwargs["data"]
    task_queue = kwargs["task_queue"]
    logger = kwargs["logger"]
    index_data_queue = kwargs["index_data_queue"]
    nfs_cluster = kwargs["nfs_cluster"]
    nfs_share = kwargs["nfs_share"]
    max_index_size = kwargs["max_index_size"]
    indexing_started_flag = kwargs["indexing_started_flag"]
    index_data_count = kwargs["index_data_count"]
    index_msg_count = kwargs["index_msg_count"]
    index_data_queue_size = kwargs.get("index_data_queue_size", 10000)
    progress_of_indexing = kwargs["progress_of_indexing"]
    prefix_index_data = kwargs["prefix_index_data"]
    standalone = kwargs.get("standalone", False)
    params = kwargs["params"]
    resume_flag = params["resume_flag"]
    progress_of_indexing[dir] = 'Progress'

    if indexing_started_flag.value == 0:
        indexing_started_flag.value = 1
        logger.info('Indexing on the shares started')

    if not os.access(dir, os.R_OK):
        logger.error('Read permission failed on dir - {}'.format(dir))
        progress_of_indexing.pop(dir)
        return

    # Reset the counters as we are going to re-index on the directories that are not completely done in the
    # previous run
    prefix_dir = dir[1:] + '/'
    if prefix_dir in prefix_index_data:
        if "files" not in prefix_index_data[prefix_dir]:
            prefix_index_data[prefix_dir] = {"files": 0, "size": 0}

    for result in iterate_dir(data=dir, task_queue=task_queue, logger=logger, max_index_size=max_index_size,
                              resume_flag=resume_flag):
        if "dir" not in result:
            logger.error("Something went wrong with the iterate_dir - {}".format(str(result)))
            continue
        # If no files in the directory, then create a task
        # Otherwise push the files list to index data queue
        if "files" in result:
            if "size" in result and result["size"] == 0:
                logger.info("No files in dir {}".format(result["dir"]))
                continue

            # Check whether the operation on the directory is already completed or not
            prefix_dir = result["dir"][1:] + '/'
            if prefix_dir not in prefix_index_data:
                prefix_index_data[prefix_dir] = dict()
                prefix_index_data[prefix_dir] = {"files": 0, "size": 0}

            msg = {"dir": result["dir"], "files": result["files"], "size": result["size"],
                   "nfs_cluster": nfs_cluster,
                   "nfs_share": nfs_share}
            try:
                # Don't let index_data_queue grow more than given value or the default 5K
                while index_data_queue.qsize() > index_data_queue_size:
                    # logger.info("Index data Q threshold crossed, Size = {}, Threshold = {}".format(index_data_queue.qsize(), index_data_queue_size))
                    time.sleep(0.1)
                if standalone:
                    data = {"dir": msg["dir"], "files": msg["files"], "size": msg["size"]}
                    logger.debug('Created Task for PUT for data-{}'.format(data))
                    task = Task(operation='put',
                                data=data,
                                s3config=params['s3config'],
                                dryrun=params['dryrun'])
                    task_queue.put(task)
                else:
                    index_data_queue.put(msg)
                    with index_msg_count.get_lock():
                        index_msg_count.value += 1
            except Exception as e:
                logger.excep("Not able to enqueue index-msg : {}".format(e))

            try:
                with index_data_count.get_lock():
                    file_count = len(result["files"])
                    index_data_count.value += file_count
                    if not resume_flag:
                        data_dict = prefix_index_data[prefix_dir]
                        data_dict["files"] += file_count
                        data_dict["size"] += result["size"]
                        prefix_index_data[prefix_dir] = data_dict
            except Exception as e:
                logger.error('Error in updating the prefix_index_data during indexing for dir {}'.format(
                    result['dir']))

            logger.debug("Index-Data_Queue:MSG= Dir-{}, Files-{}, Size-{}".format(
                result["dir"], len(result["files"]), index_data_queue.qsize()))
        else:
            subdir = result['dir']
            if subdir in progress_of_indexing and progress_of_indexing[subdir] in ['Pending', 'Progress']:
                logger.warn('SKIPPING the dir {} already present'.format(subdir))
                continue
            progress_of_indexing[subdir] = 'Pending'
            task = Task(operation="indexing", data=subdir, nfs_cluster=nfs_cluster, nfs_share=nfs_share,
                        s3config=params['s3config'], dryrun=params['dryrun'],
                        max_index_size=max_index_size, resume_flag=resume_flag)
            task_queue.put(task)

    progress_of_indexing.pop(dir)


def indexing(**kwargs):
    """
    Create a metadata index during PUT operation only.
    First Phase:
    - Create a JOSN structure for each NFS share with a NFS share name.
      Required information - directory name, file name, counts, total size of directory.

    :return:
    """
    logger = kwargs["logger"]
    params = kwargs["params"]
    dir_prefixes_to_resume = params.get("dir_prefixes_to_resume", None)
    resume_flag = params["resume_flag"]

    if resume_flag:
        logger.info('Indexing in RESUME mode')
        logger.debug('Indexing the data using the resume directories {}'.format(dir_prefixes_to_resume))
        for prefix in dir_prefixes_to_resume:
            dir_name = os.path.abspath('/' + prefix)
            kwargs['data'] = dir_name
            indexing_dir(**kwargs)
    else:
        logger.debug('Indexing the directory {}'.format(kwargs['data']))
        indexing_dir(**kwargs)


def indexing_with_file_counters(**kwargs):
    """
    Create a metadata index during PUT operation only.
    First Phase:
    - Create a JOSN structure for each NFS share with a NFS share name.
      Required information - directory name, file name, counts, total size of directory.

    :return:
    """
    # print("Actually at indexing function .... {}".format(kwargs))
    dir = kwargs["data"]
    task_queue = kwargs["task_queue"]
    logger = kwargs["logger"]
    index_data_queue = kwargs["index_data_queue"]
    nfs_cluster = kwargs["nfs_cluster"]
    nfs_share = kwargs["nfs_share"]
    max_index_size = kwargs["max_index_size"]

    progress_of_indexing = kwargs["progress_of_indexing"]
    progress_of_indexing_lock = kwargs["progress_of_indexing_lock"]
    index_data_count = kwargs["index_data_count"]

    is_dir_only_consist_files = True

    if dir not in progress_of_indexing:
        progress_of_indexing_lock.acquire()
        progress_of_indexing[dir] = 0
        progress_of_indexing_lock.release()

    for result in iterate_dir(data=dir, task_queue=task_queue, logger=logger, max_index_size=max_index_size):

        # If files a directory, then create a task
        if "dir" in result and "files" in result:
            # print("Received Files: {}".format(result))
            msg = {"dir": result["dir"], "files": result["files"], "size": result["size"], "nfs_cluster": nfs_cluster,
                   "nfs_share": nfs_share}
            index_data_queue.put(msg)
            logger.debug("Index-Data_Queue:MSG= Dir-{}, Files-{}, Size-{}".format(result["dir"], len(result["files"]),
                                                                                  index_data_queue.qsize()))
            with index_data_count.get_lock():
                index_data_count.value += len(result["files"])

        elif "dir" in result:
            task = Task(operation="indexing", data=result["dir"], nfs_cluster=nfs_cluster, nfs_share=nfs_share,
                        max_index_size=max_index_size)
            task_queue.put(task)
            is_dir_only_consist_files = False

            progress_of_indexing_lock.acquire()
            if dir in progress_of_indexing:
                progress_of_indexing[dir] += 1
            progress_of_indexing_lock.release()

    # if is_dir_only_consist_files:
    #    progress_of_indexing[dir] = 0

    # print("####### Shared Dict: dir{}:{}".format(dir,progress_of_indexing))
    if is_dir_only_consist_files:
        check_progress_of_indexing(progress_of_indexing, progress_of_indexing_lock, dir, nfs_share, logger)
    # print("============ Shared Dict: dir{}:{}".format(dir, progress_of_indexing))


def check_progress_of_indexing(progress_of_indexing, progress_of_indexing_lock, dir, nfs_share, logger):
    """
    This function helps to detect that indexing for hierarchical directory structure is completed.
        A               A=2
        |-B           B=0 C=0
        |-C
        The numeric value beside the dir name shows number of children of A. B, C are two leaf directories and
        contains only files. Hence, their value is set to 0.
        progress_of_indexing = {"/A":2,"/B":0,"/C":0}
        As B,C finishes indexing through iterator, it decrement values of its parent. The parent is found from hash key.
    :param progress_of_indexing: A shared memory contains a shared dictionary to be used by multiple processes.
    :param progress_of_indexing_lock: A lock applied for all READ/write operation on shared dict.
    :param dir: hash_key , here "/A", "/A/B", "/A/C" etc
    :param logger: A shared process safe queue.
    :return:
    """
    hierarchical_dirs = dir.split("/")
    hash_key = dir
    if hash_key in progress_of_indexing:
        try:
            key_remove_error = ""
            all_children_processed = False  # True for all parent with processed children, except root, that we don't want to delete.
            progress_of_indexing_lock.acquire()
            if hash_key in progress_of_indexing and progress_of_indexing[hash_key] > 0:
                progress_of_indexing[hash_key] -= 1

            if progress_of_indexing[hash_key] == 0 and len(dir) > len(nfs_share):
                all_children_processed = True
                # Remove child directory under NFS share
                key_remove_error = progress_of_indexing.pop(hash_key, "KEY_DOESNOT_EXIST")

            progress_of_indexing_lock.release()
            # To check if all the children processed for a prefix dir, Should stop when it reaches to NFS share root .
            if all_children_processed:
                parent_hash_key = "/".join(hierarchical_dirs[:-1])
                check_progress_of_indexing(progress_of_indexing, progress_of_indexing_lock, parent_hash_key, nfs_share,
                                           logger)

        except Exception as e:
            logger.excep("{}: {} : {}".format(__file__, e, key_remove_error))


def iterate_dir(**kwargs):
    """
    Iterate directory and return result in JSON
    :param dir:
    :return:
    """

    file_set = []
    file_set_size = 0
    file_count = 0
    dir = kwargs["data"]
    logger = kwargs["logger"]
    max_index_size = kwargs["max_index_size"]
    dm_resume = kwargs["resume_flag"]

    for entry in os.scandir(dir):
        # Check if the file a directory, then create a task
        path = entry.path
        if entry.is_dir():
            if not dm_resume:
                yield {"dir": path}
        else:
            file_size = entry.stat().st_size
            # Eliminate zero size file.
            if file_size == 0:
                #logger.warn("Zero Byte File - {}".format(entry.name))
                continue
            if file_count == max_index_size:
                yield {"dir": dir, "files": file_set, "size": file_set_size}
                file_set = [entry.name]
                file_set_size = file_size
                file_count = 1
            else:
                file_set.append(entry.name)
                file_set_size += file_size
                file_count += 1

    # Remaining files to be added.
    if file_set:
        yield {"dir": dir, "files": file_set, "size": file_set_size}
