
import os,sys
from utils.utility import exception
from multiprocessing import Value,Manager
from minio_client import MinioClient

task_id = Value('i', 1)

mgr = Manager()
ds = mgr.dict()



"""
Need to be updated 
"""
@exception
def upload(params, status_queue , logger_queue):
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
def list(index, logger_queue):
    for i in index:
        logger_queue.put("List files for object keys - {}".format(i))
@exception
def get():
    print("Download functionality ")
@exception
def delete():
    print("Remove files")


class Task:
  def __init__(self,**kwargs):
    self.id = task_id.value
    task_id.value +=1
    self.operation = kwargs["operation"]
    kwargs.pop("operation")
    self.params = kwargs
    #self.data = kwargs["data"]

  def start(self, **queue):
    self.logger_queue = queue["logger_queue"]
    self.task_queue   = queue["task_queue"]
    self.index_data_queue = queue["index_data_queue"]

    #self.logger_queue.put("====>>INFO: TASK: Started task ...!")
    #print("====>>INFO: TASK: Started task ...! {}".format(self.params))
    try:
      self.params["logger_queue"] = self.logger_queue
      if self.operation == "upload":
        upload(self.params, queue["status_queue"], self.logger_queue)
      elif self.operation == "indexing":
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
        The numeric value beside, dir name shows number of children of A. B, C are two leaf directories and
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