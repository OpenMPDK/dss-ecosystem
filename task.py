
import os,sys
from utils.utility import exception
from multiprocessing import Value,Manager
from minio_client import MinioClient
import json

task_id = Value('i', 1)

mgr = Manager()
ds = mgr.dict()



"""
Need to be updated 
"""
@exception
def put(params, status_queue , logger_queue):
    index_data = params.get("data",{})
    s3config   = params["s3config"]
    minio_config = s3config.get("minio",{})
    minio_url = minio_config["url"]
    minio_access_key = minio_config["access_key"]
    minio_secret_key = minio_config["secret_key"]
    minio_bucket = s3config.get("bucket","bucket")

    success = 0
    #logger_queue.put("TASK:Upload Minio Config {}".format(minio_config))
    mc = MinioClient(minio_url,minio_access_key,minio_secret_key)
    #print("Uploaded files dir:{}/{}".format(index_data["dir"], index_data["files"]))
    #logger_queue.put("TASK:Uploaded files dir:{}/{}".format(index_data["dir"], index_data["files"]))
    uploaded_files = []
    if mc.client:
        for file_name in index_data["files"]:
            file = os.path.abspath(index_data["dir"] + "/" + file_name)
            if os.path.exists(file):
                if mc.put(minio_bucket, file):
                    #logger_queue.put("TASK: Uploaded file {}".format(file))
                    uploaded_files.append(file_name)
                    success +=1
            else:
                logger_queue.put("ERROR: Task-Upload: File-{} doesn't exist".format(file))
    else:
        logger_queue.put("ERROR: Unable to connect to Minio for upload with {}".format(minio_config))

    #logger_queue.put("DEBUG: Minio Uploaded files {}:{}".format(index_data["dir"], uploaded_files ))
    # Update following section for upload status.
    status_message = {"success": success, "failure": (len(index_data["files"]) - success) , "dir": index_data["dir"]}
    logger_queue.put("DEBUG: Minio Upload Status - {}".format(status_message))
    status_queue.put(status_message)

@exception
def list(params, task_queue,index_data_queue, logger_queue, listing_progress, listing_progress_lock):
    max_index_size = params["max_index_size"]

    prefix_index_data_file = "/var/log/prefix_index_data.json"
    with open(prefix_index_data_file, "r") as prefix_index_data_handler:
        prefix_index_data = json.load(prefix_index_data_handler)
    #print(prefix_index_data)
    prefix = None
    if params.get("data", {}) and params["data"].get("prefix", None):
        prefix = (params["data"]["prefix"]).strip()

    # That mean lowest level of directory.
    s3config = params["s3config"]
    minio_config = s3config.get("minio", {})
    minio_url = minio_config["url"]
    minio_access_key = minio_config["access_key"]
    minio_secret_key = minio_config["secret_key"]
    minio_bucket = s3config.get("bucket", "bucket")

    mc = MinioClient(minio_url, minio_access_key, minio_secret_key)

    #if prefix not in listing_progress:
    #    listing_progress[prefix] = 0

    if prefix in prefix_index_data:
        #print("*****Prefix:{}".format(prefix))
        object_keys_iterator = mc.list(minio_bucket, prefix)
        if object_keys_iterator:
            lowest_level_directory= True
            for result in list_object_keys(object_keys_iterator, prefix_index_data,max_index_size):
                if "object_keys" in result:
                    index_data_message ={"dir": prefix, "files": result["object_keys"]}
                    #print("=====>>> TASK-INFO: Index Data Message - {}".format(index_data_message))
                    index_data_queue.put(index_data_message)
                else:
                    task = Task(operation="list", data=result, s3config=params["s3config"], max_index_size=max_index_size)
                    task_queue.put(task)
                    # Keep track of progress of listing
                    lowest_level_directory = False
                    if prefix in listing_progress:
                        listing_progress[prefix] +=1
                    else:
                        listing_progress[prefix] = 1
            if lowest_level_directory:
                listing_progress[prefix] = 0
                check_listing_progress(listing_progress,listing_progress_lock,prefix, logger_queue)
        else:
            logger_queue.put("ERROR: No object keys belongs to the prefix-{}".format(prefix))
    else:
        object_keys = mc.list(minio_bucket, prefix)
        if object_keys:
            for obj_key in object_keys:
                if prefix in listing_progress:
                    listing_progress[prefix] += 1
                else:
                    listing_progress[prefix] = 1
                task = Task(operation="list", data={"prefix":obj_key.object_name}, s3config=params["s3config"], max_index_size=max_index_size)
                task_queue.put(task)
        else:
            logger_queue.put("ERROR: No object keys belongs to the prefix-{}".format(prefix))
            print("ERROR: No object keys belongs to the prefix-{}".format(prefix))



def list_object_keys(object_keys_iterator, prefix_index_data, max_index_size):

    object_keys = []
    for obj_key in object_keys_iterator:
        # To handle a scenario in which directory has both file and directory
        if obj_key.object_name in prefix_index_data:
            yield {"prefix": obj_key.object_name}
        else:
            if len(object_keys) == max_index_size:
                #print("++++++=============>ObjectKey-:{}".format(object_keys))
                yield {"object_keys": object_keys}
                object_keys = [obj_key.object_name]
            else:
                object_keys.append(obj_key.object_name)
    if object_keys:
        yield {"object_keys": object_keys}


def check_listing_progress(listing_progress,listing_progress_lock, prefix, logger_queue):
    if prefix in listing_progress:
        try:
            listing_progress_lock.acquire()
            if listing_progress[prefix] > 0:
                listing_progress[prefix] -= 1
            listing_progress_lock.release()
            if listing_progress[prefix] == 0 and len(prefix.split("/")) > 2:
                # Check parent node with parent_hash_key
                parent_prefix = "/".join(prefix.split("/")[:-2]) + "/"
                # Remove non-top directory
                #print("TTTTTTTTTTTTTTTT:{}=>{}".format(prefix, listing_progress[prefix]))
                listing_progress_lock.acquire()
                if parent_prefix in listing_progress:
                    del listing_progress[prefix]
                listing_progress_lock.release()

                if parent_prefix in listing_progress:
                    #print("KKKKKKKKKKK:{}=>{}".format(parent_prefix, listing_progress[parent_prefix]))
                    check_listing_progress(listing_progress,listing_progress_lock, parent_prefix, logger_queue)

        except Exception as e:
            logger_queue.put("EXCEPTION:{}: {} =={}".format(__file__, e, listing_progress))
            print("EXCEPTION:{}: {} =={} ".format(__file__, e, listing_progress))

@exception
def get():
    print("Download functionality ")
@exception
def delete(params, status_queue , logger_queue):
    #print("Remove files")
    #logger_queue.put("+++++++++++++++++++++++++++ Finally DELETE {}".format(params))
    object_keys = params.get("data", {})
    s3config = params["s3config"]
    minio_config = s3config.get("minio", {})
    minio_url = minio_config["url"]
    minio_access_key = minio_config["access_key"]
    minio_secret_key = minio_config["secret_key"]
    minio_bucket = s3config.get("bucket", "bucket")

    success = 0
    mc = MinioClient(minio_url, minio_access_key, minio_secret_key)
    removed_objects = []
    object_keys = params["data"]
    if mc.client:
        for object_key in object_keys["files"]:
            #logger_queue.put("TASK: Going to removed object key {}".format(object_key))
            if mc.delete(minio_bucket, object_key):
                #logger_queue.put("TASK: Removed object key {}".format(object_key))
                removed_objects.append(object_key)
                success += 1
    else:
        logger_queue.put("ERROR: Unable to connect to Minio for upload with {}".format(minio_config))

    #logger_queue.put("DEBUG: DELETED object keys-{}".format(removed_objects ))
    # Update following section for upload status.
    status_message = {"success": success, "failure": (len(object_keys["files"]) - success), "dir": object_keys["dir"]}
    logger_queue.put("DEBUG: Task DELETE Status - {} , STATUS Q SIZE:{}".format(status_message, status_queue.qsize()))
    status_queue.put(status_message)




class Task:
  def __init__(self,**kwargs):
    self.id = task_id.value
    task_id.value +=1
    self.operation = kwargs["operation"]
    kwargs.pop("operation")
    self.params = kwargs
    #self.data = kwargs["data"]
    #print("Params:{}".format(self.params))

  def start(self, **queue):
    self.logger_queue = queue["logger_queue"]
    self.task_queue   = queue["task_queue"]
    self.index_data_queue = queue["index_data_queue"]

    #self.logger_queue.put("====>>INFO: TASK: Started task for operation-{}".format(self.operation))
    #print("====>>INFO: TASK: Started task ...! {}".format(self.params))
    try:
      self.params["logger_queue"] = self.logger_queue
      if self.operation.lower() == "put":
        put(self.params, queue["status_queue"], self.logger_queue)
      elif self.operation.lower() == "list":
        list(self.params, queue["task_queue"], queue["index_data_queue"] , self.logger_queue,queue["listing_progress"], queue["listing_progress_lock"])
      elif self.operation.lower() == "del":
        delete(self.params, queue["status_queue"], self.logger_queue)
      elif self.operation.lower() == "indexing":
        indexing(data=self.params["data"],
                 nfs_cluster=self.params["nfs_cluster"],
				 task_queue=queue["task_queue"],
				 index_data_queue=queue["index_data_queue"],
                 logger_queue=queue["logger_queue"],
                 progress_of_indexing=queue["progress_of_indexing"],
                 progress_of_indexing_lock=queue["progress_of_indexing_lock"],
                 max_index_size=self.params.get("max_index_size", 10)
                 )

    except Exception as e:
        self.logger_queue.put("Exception:{}:Task: {}!".format(__file__,e))
        print("Exception:{}:{}:{}".format(__file__,e, self.params["data"] ))

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
    #print("Actually at indexing function .... {}".format(kwargs))
    dir = kwargs["data"]
    task_queue = kwargs["task_queue"]
    logger_queue = kwargs["logger_queue"]
    index_data_queue = kwargs["index_data_queue"]
    nfs_cluster = kwargs["nfs_cluster"]
    max_index_size= kwargs["max_index_size"]

    progress_of_indexing = kwargs["progress_of_indexing"]
    progress_of_indexing_lock = kwargs["progress_of_indexing_lock"]

    is_dir_only_consist_files = True

    if dir not in progress_of_indexing:
        progress_of_indexing_lock.acquire()
        progress_of_indexing[dir] = 0
        progress_of_indexing_lock.release()

    for result in iterate_dir(data=dir, task_queue=task_queue, logger_queue=logger_queue, max_index_size=max_index_size):

        # If files a directory, then create a task
        if "dir" in result and "files" in result:
            #print("Received Files: {}".format(result["files"]))
            msg = {"dir": result["dir"], "files": result["files"] , "nfs_cluster": nfs_cluster}
            #print("Index-Data_Queue:MSG={}".format(msg))
            index_data_queue.put(msg)


        elif "dir" in result:
            task = Task(operation="indexing", data=result["dir"], nfs_cluster=nfs_cluster, max_index_size=max_index_size )
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
        check_progress_of_indexing(progress_of_indexing,progress_of_indexing_lock ,dir, logger_queue)
    #print("============ Shared Dict: dir{}:{}".format(dir, progress_of_indexing))


def check_progress_of_indexing(progress_of_indexing, progress_of_indexing_lock,  dir, logger_queue):
    """
    This function helps to detect that indexing for hierarchical directory structure is completed.
        A               A=2
        |-B           B=0 C=0
        |-C
        The numeric value beside the dir name shows number of children of A. B, C are two leaf directories and
        contains only files. Hence, their value is set to 0.
        preogress_of_indexing = {"/A":2,"/B":0,"/C":0}
        As B,C finishes indexing through iterator, it decrement values of its parent. The parent is found from hash key.
    :param progress_of_indexing: A shared memory contains a shared dictionary to be used by multiple processes.
    :param progress_of_indexing_lock: A lock applied for all READ/write operation on shared dict.
    :param dir: hash_key , here "/A", "/A/B", "/A/C" etc
    :param logger_queue: A shared process safe queue.
    :return:
    """
    hierarchical_dirs = dir.split("/")
    hash_key = "/".join(hierarchical_dirs)
    if hash_key in progress_of_indexing:
        try:
            progress_of_indexing_lock.acquire()
            if progress_of_indexing[hash_key] > 0:
                progress_of_indexing[hash_key] -= 1
            progress_of_indexing_lock.release()
            if progress_of_indexing[hash_key] == 0 and len(hierarchical_dirs) > 2:
                # Remove non-top directory
                #print("TTTTTTTTTTTTTTTT:{}=>{}".format(hash_key, progress_of_indexing[hash_key]))
                progress_of_indexing_lock.acquire()
                del progress_of_indexing[hash_key]
                progress_of_indexing_lock.release()
                # Check parent node with parent_hash_key
                parent_hash_key = "/".join(hierarchical_dirs[:-1])
                check_progress_of_indexing(progress_of_indexing,progress_of_indexing_lock, parent_hash_key, logger_queue)
        except Exception as e:
            logger_queue.put("EXCEPTION:{}: {} ".format(__file__, e))
            #print("EXCEPTION:{}: {} ".format(__file__, e))

def iterate_dir(**kwargs):
    """
    Iterate directory and return result in JSON
    :param dir:
    :return:
    """
    #print("DEBUG: Iterating - {}".format(kwargs))

    #file_count=10
    file_set = []
    dir = kwargs["data"]
    logger_queue = kwargs["logger_queue"]
    max_index_size = kwargs["max_index_size"]


    for file in os.listdir(dir):
        # Check if the file a directory, then create a task
        path = os.path.abspath(dir + "/" + file)

        if os.path.isdir(path):
            yield { "dir": path }
        else:
            if len(file_set) == max_index_size:
                yield {"dir": dir, "files": file_set}
                #print("Files:{}".format(file_set))
                file_set = [file]
            else:
                file_set.append(file)

    # Remaining files to be added.
    if file_set:
        yield {"dir": dir, "files": file_set}