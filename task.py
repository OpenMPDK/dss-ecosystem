
import os,sys
from utils.utility import exception
from multiprocessing import Value
from minio_client import MinioClient

task_id = Value('i', 1)



"""
Need to be updated 
"""
@exception
def upload(index, status_queue , logger_queue):
    print("Uploaded files dir:{}/{}".format(index["dir"],index["files"]))
    logger_queue.put("===>>>TASK:Uploaded files dir:{}/{}".format(index["dir"],index["files"]))

    # Update following section for upload status.
    status_queue.put({"success": len(index["files"]), "failure": 0})

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

  def start(self, **queue):
    self.logger_queue = queue["logger_queue"]
    self.task_queue   = queue["task_queue"]
    self.index_data_queue = queue["index_data_queue"]

    self.logger_queue.put("====>>INFO: TASK: Started task ...!")
    print("====>>INFO: TASK: Started task ...! {}".format(self.params))
    try:
      self.params["logger_queue"] = self.logger_queue
      if self.operation == "upload":
        upload(self.params["data"], queue["status_queue"], self.logger_queue)
      elif self.operation == "indexing":
        indexing(data=self.params["data"],
                 nfs_cluster=self.params["nfs_cluster"],
				 task_queue=queue["task_queue"],
				 index_data_queue=queue["index_data_queue"],
                 logger_queue=queue["logger_queue"],
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

    for result in iterate_dir(data=dir, task_queue=task_queue, logger_queue=logger_queue):
        #print("Received Files: {}".format(result))
        # If files a directory, then create a task
        if "dir" in result and "files" in result:
            # else send message
            msg = {"dir": result["dir"], "files": result["files"] , "nfs_cluster": nfs_cluster}
            print("Index-Data_Queue:MSG={}".format(msg))
            index_data_queue.put(msg)

        elif "dir" in result:
            task = Task(operation="indexing", data=result["dir"], nfs_cluster=nfs_cluster )
            task_queue.put(task)

def iterate_dir(**kwargs):
    """
    Iterate directory and return result in JSON
    :param dir:
    :return:
    """
    #print("DEBUG: Iterating - {}".format(kwargs))
    index = 0
    file_count=10
    file_set = []
    dir = kwargs["data"]
    logger_queue = kwargs["logger_queue"]


    for file in os.listdir(dir):

        #print("ccdssd{}".format(file))
        # Check if the file a directory, then create a task
        path = os.path.abspath(dir + "/" + file)

        if os.path.isdir(path):
           # print("DEBUG: iterate_dir - path:{}".format(path))
            yield { "dir": path }
        else:
            #print("DEBUG:  file - path:{}".format(path))
            if len(file_set) == file_count:
                yield {"files": file_set}
                #print("Files:{}".format(file_set))
                file_set = []
            else:
                file_set.append(file)
        index +=1

    # Remaining files to be added.
    if file_set:
        yield {"dir": dir, "files": file_set}