#!/usr/bin/python

import os,sys
import boto3
from botocore.client import Config
from datetime import datetime

class S3:
  def __init__(self, config):
    self.s3_client = boto3.client('s3',
                        endpoint_url='http://202.0.0.137:9000',
                        aws_access_key_id='minio',
                        aws_secret_access_key='minio123',
                        config=Config(signature_version='s3v4'),
                        region_name='us-east-1')


  def get_s3_client(self):
    return self.s3_client

  def create_bucket(self, bucket=None):
    """
    Create a bucket into S3 storage
    :param bucket:
    :return:
    """
    if bucket:
      try:
        self.s3_client.create_bucket(Bucket=bucket)
      except Exception as e:
        print("EXCEPTION: {}".format(e))
        return False
    return True


  def upload_file(self, bucket=None, source_file_path=None):
    """
    Given a bucket and source file, upload the file to the respective bucket.
    :param bucket:
    :param source_file_path:
    :return:
    """
    if not bucket:
        print("ERROR: Requires Bucket name!")
        return False

    object_key = source_file_path[1:]
    if not object_key.startswith("/"):
        self.s3_client.Bucket(bucket).upload_file(source_file_path, object_key)

  def upload_fileobj(self, bucket=None, source_file_path=None):
    """
    Upload a file object given a object key
    :param bucket:
    :param source_file_path:
    :return:
    """

    if not bucket:
        print("ERROR: Requires bucket name")
        return False
    object_key = source_file_path[1:]
    if os.path.exists(source_file_path) and os.path.isfile(source_file_path)  and not object_key.startswith("/"):
      with open(source_file_path, "rb") as fh:
        ret = self.s3_client.upload_fileobj(Fileobj=fh,
                                            Bucket=bucket,
                                            Key=object_key)
    else:
        return False
    return True


  def list_buckets(self):
    """
    List buckets in a object storage.
    :return:
    """
    buckets = []
    try:
      response = self.s3_client.list_buckets()
      for bucket in response["Buckets"]:
        buckets.append(bucket["Name"])
    except Exception as e:
      print("EXCEPTION: {}".format(e))

    return buckets

  def list_objects(self, bucket=None, prefix=None):
    """
    List all the objects under a bucket and return
    :param bucket: Bucket name required
    :param prefix: Prefix optional, if specified then return all the object keys under that prefix
    :return:
    """
    object_keys = []
    if bucket and prefix:
      response = self.s3_client.list_objects(Bucket=bucket,Prefix=prefix)
      #for object in bucket_obj.objects.all():
      #  print(object.key)
      #  object_keys.append(object.key)
      for content in response["Contents"]:
        #print(content["Key"])
        object_keys.append(content["Key"])
        yield content["Key"]

    #return object_keys

  def download_file(self, object_key):
    # Create local directory structure
    local_file_path = "/" + object_key
    directory  =  os.path.dirname(local_file_path)
    if not os.path.exist(directory):
      os.makdir(directory)
    self.s3_client.download_file(object_key, local_file_path)

  def delete_objects(self, bucket=None,prefix=None):
    """
    Remove a object based on object key.
    :param bucket: Bucket name required
    :param prefix: prefix required
    :return:
    """
    if not bucket or not prefix:
      return False
    try:
      self.s3_client.delete_object(Bucket=bucket, Key=prefix)
    except Exception as e:
      print("EXCEPTION: {}".format(e))
      return False
    return True





if __name__ =="__main__":
    s3=  S3({})
    # Example of uploading files
    print("Uploading files")
    directory = "/bird/bird1/"
    uploaded_file_count = 0
    now = datetime.now()

    for file in os.listdir(directory):
        file_path = directory + file
        if s3.upload_fileobj("bucket", file_path):
          uploaded_file_count +=1
    print("Uploaded Files:{} , Time-{}".format(uploaded_file_count, (datetime.now() - now).seconds))

    # Example of listing buckets.
    print("All Buckets: {}".format(s3.list_buckets()))

    prefix = "bird/"
    # List object keys
    #print("List Object Keys: {}".format(s3.list_objects("bucket", "bird/")))

    # Delete object
    deleted_object_count = 0
    for object_key in s3.list_objects("bucket", prefix):
      if s3.delete_objects("bucket", object_key):
        deleted_object_count +=1
    print("Deleted Objects Count: {} for prefix-\"{}\"".format(deleted_object_count, prefix))






