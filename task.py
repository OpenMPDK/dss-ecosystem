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
from utils.utility import exception, exec_cmd, get_hash_key
from multiprocessing import Value,Manager
from minio_client import MinioClient
from s3_client import S3
import json
from datetime import datetime

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
    :param kwargs: conatins params, status_queue, logger
    :return:
    """
    params = kwargs["params"]
    status_queue = kwargs["status_queue"]
    logger = kwargs["logger"]

    index_data = params.get("data",{})
    s3config   = params["s3config"]
    minio_bucket = s3config.get("bucket","bucket")

    # Data Integrity
    data_integrity = kwargs.get("data_integrity", False)
    file_hash_map = kwargs.get("file_hash_map", {})
    dryrun = params.get("dryrun", False)

    success = 0
    failure_files_size = 0
    if s3_client:
        start_time = datetime.now()
        for file_name in index_data["files"]:
            file = os.path.abspath(index_data["dir"] + "/" + file_name)
            if os.path.exists(file):
                try:
                    if dryrun:
                        # Read file for the purpose of testing
                        with open(file, "rb") as FH:
                            lines = FH.readlines()
                        lines = []
                        success +=1
                    else:
                        if s3_client.putObject(minio_bucket, file):
                            if data_integrity:
                                file_content_hash_key = get_hash_key(type="file",
                                                                     data=file,
                                                                     logger=logger)
                                file_hash_map[file_name] = file_content_hash_key
                            success +=1
                        else:
                            failure_files_size += os.path.getsize(file)
                except Exception as e:
                    logger.excep("PUT - {}".format(e))
                    failure_files_size += os.path.getsize(file)
            else:
                logger.error("Task-Upload: File-{} doesn't exist".format(file))
        upload_time = (datetime.now() - start_time).seconds
        logger.debug("Upload Time: {} sec".format(upload_time))
    else:
        logger.error("Unable to connect to S3 Storage for upload")

    if data_integrity:
        logger.debug("HashMap-(FileName->HashKey): {}".format(file_hash_map))
    else:
        # Update following section for upload status.
        status_message = {"success": success, "failure": (len(index_data["files"]) - success) ,
                          "dir": index_data["dir"] ,
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
    :param listing_progress: it holds the
    :param listing_progress_lock:
    :return:
    """
    params = kwargs["params"]
    logger = kwargs["logger"]
    task_queue = kwargs["task_queue"]  # Contains task Object
    index_data_queue = kwargs["index_data_queue"] # Contains index_data
    index_data_count = kwargs["index_data_count"]
    listing_progress = kwargs["listing_progress"] # The shared memory is used to hold progress status.
    listing_progress_lock = kwargs["listing_progress_lock"]
    listing_status = kwargs["listing_status"]
    listing_only = kwargs["listing_only"]
    listing_objectkey_queue = kwargs["listing_objectkey_queue"]

    max_index_size = params["max_index_size"]

    prefix_index_data_file = "/var/log/prefix_index_data.json"
    with open(prefix_index_data_file, "r") as prefix_index_data_handler:
        prefix_index_data = json.load(prefix_index_data_handler)

    prefix = None
    if params.get("data", {}) and params["data"].get("prefix", None):
        prefix = (params["data"]["prefix"]).strip()

    # That mean lowest level of directory.
    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")
    s3_client_library = s3config.get("client_lib","minio_client")
    logger.debug("Performing LISTing with prefix - {}".format(prefix))

    if prefix in prefix_index_data:
        object_keys_iterator = s3_client.listObjects(minio_bucket, prefix)
        if object_keys_iterator:
            for result in list_object_keys(object_keys_iterator, max_index_size, s3_client_library):
                if "object_keys" in result:
                    index_data_message ={"dir": prefix, "files": result["object_keys"]}
                    object_keys_count = len(result["object_keys"])
                    with index_data_count.get_lock():
                        index_data_count.value += object_keys_count
                    logger.debug("LIST: Prefix-\"{}\" ObjectKeys: {}".format(prefix, object_keys_count)) ## DELETE
                    if listing_only.value:
                      result.update({"prefix" : prefix})
                      listing_objectkey_queue.put(result)
                    else:
                      index_data_queue.put(index_data_message)
                else:
                    listing_progress_lock.acquire()
                    listing_progress[result["prefix"]] = 1
                    listing_progress_lock.release()
                    task = Task(operation="list", data=result, s3config=params["s3config"], max_index_size=max_index_size)
                    task_queue.put(task)
        else:
            logger.error("No object keys belongs to the prefix-{}".format(prefix))
    else:
        object_keys = s3_client.listObjects(minio_bucket, prefix)
        if object_keys:
            for obj_key in object_keys:
                if s3_client_library.lower() == "minio":
                    object_key_prefix =  obj_key.object_name
                elif s3_client_library.lower() == "dss_client":
                    object_key_prefix =  obj_key
                listing_progress_lock.acquire()
                listing_progress[object_key_prefix] = 1
                listing_progress_lock.release()
                task = Task(operation="list", data={"prefix":object_key_prefix}, s3config=params["s3config"], max_index_size=max_index_size)
                task_queue.put(task)
        else:
            logger.error("No object keys belongs to the prefix-{}".format(prefix))

    if listing_status.value  != 1:
        listing_status.value = 1
    #listing_progress[prefix] = 0
    listing_progress_lock.acquire()
    if prefix in listing_progress:
      del listing_progress[prefix]
    listing_progress_lock.release()



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
def get(s3_client,**kwargs):
    params = kwargs["params"]
    status_queue = kwargs["status_queue"]
    logger = kwargs["logger"]
    dest_path = params.get("dest_path","")
    user_id = params["user_id"]

    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")
    dryrun = params.get("dryrun", False)

    # Data Integrity
    data_integrity = kwargs.get("data_integrity", False)
    file_hash_map = kwargs.get("file_hash_map", {})
    success = 0
    object_keys = params["data"]
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
        except Exception as e:
          logger.excep("Prefix-{}\n{}".format(object_keys["dir"], e))

        if ret == 0:
            for object_key in object_keys["files"]:
                if dryrun:  # Dry run
                    success += 1
                else: # Actual operation
                    dest_file_path = dest_path + "/" + object_key
                    # logger.debug("Obj_Key:{}, Dest_Path:{}".format(object_key,dest_file_path))
                    if s3_client.getObject(minio_bucket, object_key, dest_file_path ):
                        if data_integrity:
                            hash_key = get_hash_key(type='file', data=dest_file_path, logger=logger)
                            file_name = object_key.split("/")[-1]
                            if file_hash_map[file_name] == hash_key:
                                success +=1
                            else:
                                logger.error("Failed DataIntegrity for Object Key - {}".format(object_key))
                        else:
                            success += 1
                    else:
                        logger.error("Failed to download object - {}".format(object_key))
    else:
        logger.error("Unable to connect to S3 ObjectStorage for download")

    # Update following section for upload status.
    status_message = {"success": success, "failure": (len(object_keys["files"]) - success), "dir": object_keys["dir"]}
    logger.debug("STATUS:{}".format(status_message))  ## Delete
    status_queue.put(status_message)

@exception
def delete(s3_client,**kwargs):
    params = kwargs["params"]
    status_queue = kwargs["status_queue"]
    logger = kwargs["logger"]

    s3config = params["s3config"]
    minio_bucket = s3config.get("bucket", "bucket")

    success = 0
    object_keys = params["data"]
    prefix =  object_keys["dir"]
    if s3_client:
        for object_key in object_keys["files"]:
            object_key = prefix + object_key
            #logger.debug("TASK: Going to removed object key {}".format(object_key))
            if params.get("dryrun", False):
                # Dry run
                success +=1
            else:
                # Actual operation
                if s3_client.deleteObject(minio_bucket, object_key):
                    #logger.debug("TASK: Removed object key {}".format(object_key))
                    success += 1
    else:
        logger.error("Unable to connect to Minio for upload with {}".format(minio_config))

    # Update following section for upload status.
    status_message = {"success": success, "failure": (len(object_keys["files"]) - success), "dir": prefix}
    status_queue.put(status_message)

@exception
def data_integrity(s3_client,**kwargs):
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
    # Upload first
    data_integrity = True
    file_hash_map = {}
    params = kwargs.get("params", {})
    data = params["data"]
    user_id = params["user_id"]
    status_queue = kwargs["status_queue"]
    logger.info("Checking DataIntegrity for the prefix - {}".format(data["dir"]))
    logger.info("** DataIntegrity: Performing PUT operation - Prefix-{} **".format(data["dir"]))
    put(s3_client, params=params,
                   status_queue=status_queue,
                   data_integrity=data_integrity,
                   file_hash_map=file_hash_map,
                   logger=logger)
    ## Download
    # - Fine tune the parameters for GET
    # Update data with object keys as expected by GET function
    uploaded_data = {"dir" : data["dir"], "files":[]}
    for file_name in data["files"]:
        object_key = os.path.abspath(data["dir"] + "/" + file_name)
        object_key = object_key[1:]
        uploaded_data["files"].append(object_key)
    params["data"] = uploaded_data
    logger.info("** DataIntegrity: Performing GET operation - Prefix-{} **".format(data["dir"]))
    get(s3_client, params=params,
                   status_queue=status_queue,
                   data_integrity=data_integrity,
                   file_hash_map=file_hash_map,
                   logger=logger)

    # Remove the local files
    download_path = params["dest_path"] + data["dir"]
    logger.info("DataIntegrity: Removing all the files under the download path - {}".format(download_path))
    if os.path.exists(download_path):
      command = "rm -rf {}".format(download_path)
      ret,console = exec_cmd(command, True, True, user_id)
      if ret == 0:
        logger.info("Removed all the downloaded files under prefix - {}".format(data["dir"]))
      else:
        logger.error("Failed to removed downloaded files for the prefix-{}".format(data["dir"]))



class Task:
  def __init__(self,**kwargs):
    self.id = task_id.value
    task_id.value +=1
    self.operation = kwargs["operation"]
    kwargs.pop("operation")
    self.params = kwargs

  def start(self, **queue):
    self.logger = queue["logger"]
    self.task_queue   = queue["task_queue"]
    self.index_data_queue = queue["index_data_queue"]

    s3_client = queue["s3_client"]
    # self.logger.debug("Task: Operation-{}, Data-{}".format(self.operation, self.params["data"]))
    try:
      if self.operation.lower() == "put":
        put(s3_client, params=self.params,
                       status_queue=queue["status_queue"],
                       logger=self.logger)
      elif self.operation.lower() == "list":
        list(s3_client, params=self.params,
                        task_queue=queue["task_queue"],
                        index_data_queue=queue["index_data_queue"] ,
                        index_data_count=queue["index_data_count"],
                        logger=self.logger,
                        listing_progress=queue["listing_progress"],
                        listing_progress_lock=queue["listing_progress_lock"],
                        listing_status=queue["listing_status"],
                        listing_only=queue["listing_only"],
                        listing_objectkey_queue=queue["listing_objectkey_queue"])
      elif self.operation.lower() == "del":
        delete(s3_client, params=self.params,
                          status_queue=queue["status_queue"],
                          logger=self.logger)

      elif self.operation.lower() == "get":
        get(s3_client, params=self.params,
                       status_queue=queue["status_queue"],
                       logger=self.logger)
      elif self.operation.lower() == "test":
        data_integrity(s3_client, params=self.params,
                status_queue=queue["status_queue"],
                logger=self.logger)
      elif self.operation.lower() == "indexing":
        indexing(data=self.params["data"],
                 nfs_cluster=self.params["nfs_cluster"],
                 nfs_share=self.params["nfs_share"],
                 task_queue=queue["task_queue"],
                 index_data_queue=queue["index_data_queue"],
                 logger=self.logger,
                 progress_of_indexing=queue["progress_of_indexing"],
                 progress_of_indexing_lock=queue["progress_of_indexing_lock"],
                 index_data_count=queue["index_data_count"],
                 max_index_size=self.params.get("max_index_size", 10),
                 indexing_started_flag=queue['indexing_started_flag']
                 )

    except Exception as e:
        self.logger.excep("{}:Task: {}!".format(__file__,e))

  def stop(self):
    pass


def indexing(**kwargs):
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

    progress_of_indexing = kwargs["progress_of_indexing"]
    # progress_of_indexing_lock = kwargs["progress_of_indexing_lock"]
    if dir not in progress_of_indexing:
        logger.warn('Directory {} not present in index'.format(dir))
    elif progress_of_indexing[dir] != 'Pending':
        logger.warn('Directory {} is not in pending state - {}'.format(dir, progress_of_indexing[dir]))

    progress_of_indexing[dir] = 'Progress'
    if indexing_started_flag.value == 0:
        indexing_started_flag.value = 1
        logger.info('Indexing on the shares started')
    for result in iterate_dir(data=dir, task_queue=task_queue, logger=logger,
                              max_index_size=max_index_size):
        # If files a directory, then create a task
        if "dir" in result and "files" in result:
            msg = {"dir": result["dir"], "files": result["files"], "size": result["size"],
                   "nfs_cluster": nfs_cluster,
                   "nfs_share": nfs_share}
            index_data_queue.put(msg)
            with index_data_count.get_lock():
                index_data_count.value += len(result["files"])
            logger.debug("Index-Data_Queue:MSG= Dir-{}, Files-{}, Size-{}".format(
                result["dir"], len(result["files"]), index_data_queue.qsize()))
        elif "dir" in result:
            subdir = result['dir']
            if subdir in progress_of_indexing and progress_of_indexing[subdir] in ['Pending', 'Progress']:
                logger.warn('SKIPPING the dir {} already present'.format(subdir))
                continue
            progress_of_indexing[subdir] = 'Pending'
            task = Task(operation="indexing", data=subdir, nfs_cluster=nfs_cluster, nfs_share=nfs_share,
                        max_index_size=max_index_size)
            task_queue.put(task)

    progress_of_indexing.pop(dir)


def indexing_with_file_counters(**kwargs):
    """
    Create a metadata index during PUT operation only.
    First Phase:
    - Create a JOSN structure for each NFS share with a NFS share name.
      Required information - directory name, file name, counts, total size of directory.

    :return:
    """
    #print("Actually at indexing function .... {}".format(kwargs))
    dir = kwargs["data"]
    task_queue = kwargs["task_queue"]
    logger = kwargs["logger"]
    index_data_queue = kwargs["index_data_queue"]
    nfs_cluster = kwargs["nfs_cluster"]
    nfs_share = kwargs["nfs_share"]
    max_index_size= kwargs["max_index_size"]

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
            #print("Received Files: {}".format(result))
            msg = {"dir": result["dir"], "files": result["files"] , "size": result["size"], "nfs_cluster": nfs_cluster, "nfs_share": nfs_share}
            index_data_queue.put(msg)
            logger.debug("Index-Data_Queue:MSG= Dir-{}, Files-{}, Size-{}".format( result["dir"], len(result["files"]) , index_data_queue.qsize()))
            with index_data_count.get_lock():
                index_data_count.value += len(result["files"])

        elif "dir" in result:
            task = Task(operation="indexing", data=result["dir"], nfs_cluster=nfs_cluster, nfs_share=nfs_share, max_index_size=max_index_size )
            task_queue.put(task)
            is_dir_only_consist_files = False

            progress_of_indexing_lock.acquire()
            if dir  in progress_of_indexing:
                progress_of_indexing[dir] +=1
            progress_of_indexing_lock.release()

    #if is_dir_only_consist_files:
    #    progress_of_indexing[dir] = 0


    #print("####### Shared Dict: dir{}:{}".format(dir,progress_of_indexing))
    if is_dir_only_consist_files:
        check_progress_of_indexing(progress_of_indexing,progress_of_indexing_lock ,dir, nfs_share, logger)
    #print("============ Shared Dict: dir{}:{}".format(dir, progress_of_indexing))


def check_progress_of_indexing(progress_of_indexing, progress_of_indexing_lock,  dir, nfs_share, logger):
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
            if hash_key in progress_of_indexing and  progress_of_indexing[hash_key] > 0:
                progress_of_indexing[hash_key] -= 1

            if progress_of_indexing[hash_key] == 0 and len(dir) > len(nfs_share):
                all_children_processed = True
                # Remove child directory under NFS share
                key_remove_error = progress_of_indexing.pop(hash_key,"KEY_DOESNOT_EXIST")

            progress_of_indexing_lock.release()
            # To check if all the children processed for a prefix dir, Should stop when it reaches to NFS share root .
            if all_children_processed  :
                parent_hash_key = "/".join(hierarchical_dirs[:-1])
                check_progress_of_indexing(progress_of_indexing,progress_of_indexing_lock, parent_hash_key, nfs_share, logger)

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
    dir = kwargs["data"]
    logger = kwargs["logger"]
    max_index_size = kwargs["max_index_size"]

    for entry in os.scandir(dir):
        # Check if the file a directory, then create a task
        path = entry.path
        if entry.is_dir():
            yield {"dir": path}
        else:
            if len(file_set) == max_index_size:
                yield {"dir": dir, "files": file_set, "size": file_set_size}
                # print("Files:{}".format(file_set))
                file_set = [entry.name]
                file_set_size = entry.stat().st_size
            else:
                file_set.append(entry.name)
                file_set_size += entry.stat().st_size

    # Remaining files to be added.
    if file_set:
        yield {"dir": dir, "files": file_set, "size": file_set_size}
