import os,sys
from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists
from datetime import datetime

class MinioClient:
  def __init__(self, url, access_key="minio", secret_key="minio123"):
    self.minio_url = url
    self.client = self.get_client(url,access_key, secret_key)

  def get_client(self,url,access_key, secret_key):
    #print("Minio Server: http://{}, Access Key:{} , Secret Key:{}".format(url, access_key, secret_key))
    mc = Minio(url,access_key, secret_key, secure=False)
    return mc

  def make_bucket(self, bucket=None):
    if bucket:
      print("INFO: Bucket Name - {}".format(bucket))
      try:
        self.client.make_bucket(bucket)
        print("INFO: New bucket created - {}".format(bucket))
      except BucketAlreadyOwnedByYou as err:
        print("INFO: Bucket {} already own by you".format(bucket))
      except BucketAlreadyExists as err:
        print("WARNING: Bucket {} already exist".format(bucket))
      except err:
        print("Exception:{}".format(err))

  def list_bucket(self):
    for bucket in self.client.list_buckets():
      print("{}".format(bucket.name))

  def list(self, bucket=None, prefix=None, recursive=False):
    try:
      objects = self.client.list_objects(bucket_name=bucket,prefix=prefix,recursive=recursive)
    except err:
      print("Exception:{}".format(err))

    return objects


  def put(self, bucket=None, data=None):
    #print("Bucket Name:{}=>{}".format(bucket,data))
    try:
      if os.path.isdir(data):
        # Make recursive copy
        #print("Directory Data:{}".format(data))
        for file in os.listdir(data):
          file_path = os.path.abspath(data + "/" + file)
          print(file_path)
          self.put(bucket,file_path)
      else:
        # File copy
        #print("Data:{}".format(data))
        self.client.fput_object(bucket_name=bucket,object_name=data[1:],file_path=data,content_type="text/plain")
    except err:
      print("Exception:{}".format(err))
      return False
    return True
  def get(self, bucket=None ):
    pass


  def delete(self, bucket=None, prefix=None, recursive=False):
    try:
      if bucket:
        """
        objects = self.list(bucket,prefix,recursive)
        #self.client.remove_objects(bucket_name=bucket,objects_iter=objects)
        for object in objects:
          print("Removing object - {}".format(object.object_name))
          self.client.remove_object(bucket_name=bucket,object_name=object.object_name)  # Require object name,
        """
        self.client.remove_object(bucket_name=bucket, object_name=prefix)
    except err:
      print("Exception:{}".format(err))
      return False
    return True




if __name__=="__main__":
  mc = MinioClient("202.0.0.137:9000")
  # Example of uploading files
  print("Uploading files")
  directory = "/bird/bird1/"
  uploaded_file_count = 0
  now = datetime.now()

  for file in os.listdir(directory):
    file_path = directory + file
    if mc.put("bucket", file_path):
      uploaded_file_count += 1
  print("Uploaded Files:{} , Time-{}".format(uploaded_file_count, (datetime.now() - now).seconds))

  # Example of listing buckets.
  #print("All Buckets: {}".format(mc.list()))

  prefix = "bird/bird1/"
  # List object keys
  # print("List Object Keys: {}".format(s3.list_objects("bucket", "bird/")))

  # Delete object
  deleted_object_count = 0
  for object in mc.list("bucket", prefix):
    if mc.delete("bucket", object.object_name):
      deleted_object_count += 1
  print("Deleted Objects Count: {} for prefix-\"{}\"".format(deleted_object_count, prefix))
