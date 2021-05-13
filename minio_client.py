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
from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists
from datetime import datetime
import json

class MinioClient:
  def __init__(self, url, access_key="minio", secret_key="minio123", logger=None):
    self.minio_url = url
    self.client = self.get_client(url,access_key, secret_key)
    self.logger = logger

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

  def listObjects(self, bucket=None, prefix=None, recursive=False):
    objects = []
    try:
      if bucket:
        objects = self.client.list_objects(bucket_name=bucket,prefix=prefix,recursive=recursive)
    except err:
      self.logger.exception("{}".format(err))

    return objects


  def putObject(self, bucket=None, data=None):
    #print("Bucket Name:{}=>{}".format(bucket,data))
    try:
      self.client.fput_object(bucket_name=bucket,object_name=data[1:],file_path=data,content_type="text/plain")
    except err:
      self.logger.exception("{}".format(err))
      return False
    return True
  def get(self, bucket=None ):
    pass


  def deleteObject(self, bucket=None, prefix=None, recursive=False):
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
      self.logger.exception("{}".format(err))
      return False
    return True

  def getObject(self,bucket=None, object_key="", dest_path=""):
    """
    Download the object based on the specified object key to the specified file path.
    :param bucket: A bucket name is required and that has to be present into minio.
    :param object_key: Required
    :param dest_path: Required , destination file path
    :return:
    """
    if bucket and object_key and dest_path:
      try:
        self.client.fget_object(bucket, object_key, dest_path)
        return True
      except Exception as e:
        self.logger.exception("{}".format(e))
    return False

if __name__=="__main__":
  mc = MinioClient("202.0.0.103:9000")
  # Example of uploading files
  """
  print("Uploading files")
  directory = "/dir1"
  uploaded_file_count = 0
  now = datetime.now()

  for file in os.listdir(directory):
    file_path = directory + file
    if mc.put("bucket", file_path):
      uploaded_file_count += 1
  print("Uploaded Files:{} , Time-{}".format(uploaded_file_count, (datetime.now() - now).seconds))
  """

  # Example of listing buckets.
  #print("All Buckets: {}".format(mc.list()))

  prefix = "dir1/"
  # List object keys
  # print("List Object Keys: {}".format(s3.list_objects("bucket", "bird/")))

  # Delete object
  deleted_object_count = 0
  get_count = 0
  for object in mc.listObjects("bucket", prefix):
    if mc.getObject("bucket", object.object_name, "/home/somnath.s/work/Testing" ):
      get_count +=1
    #if mc.deleteObject("bucket", object.object_name):
    #  deleted_object_count += 1

  #print("Deleted Objects Count: {} for prefix-\"{}\"".format(deleted_object_count, prefix))
