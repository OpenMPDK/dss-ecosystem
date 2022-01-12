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
import os,sys
import boto3
from botocore.client import Config
from datetime import datetime

class S3:
  def __init__(self, config={}):
      self.storage_name = config.get("name")
      self.config = config["credentials"]
      self.get_s3_client()

  def get_s3_client(self):
      if self.storage_name.lower() == "aws":
          self.s3_client = boto3.client('s3',
                        aws_access_key_id=self.config["access_key"],
                        aws_secret_access_key=self.config["secret_key"],
                        config=Config(signature_version='s3v4'),
                        region_name='us-east-1')
      elif self.storage_name.lower() == "dss":
          self.s3_client = boto3.client('s3',
                                        endpoint_url=self.config["endpoint"],
                                        aws_access_key_id=self.config["access_key"],
                                        aws_secret_access_key=self.config["secret_key"])
      else:
          print("ERROR: Wrong object storage!")

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

  def putObject(self, bucket=None, source_file_path=None):
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

  def listObjects(self, bucket=None, prefix=None):
      """
      List all the objects under a bucket and return
      :param bucket: Bucket name required
      :param prefix: Prefix optional, if specified then return all the object keys under that prefix
      :return:
      """

      if bucket and prefix:
          try:
              paginator = self.s3_client.get_paginator('list_objects')
              params = {"Bucket": bucket, "Prefix": prefix}
              page_iterator = paginator.paginate(**params)
              for page in page_iterator:
                  object_keys = []
                  for content in page["Contents"]:
                      object_keys.append(content["Key"])
                  yield object_keys
          except Exception as e:
              print(e)


  def getObject(self, **kwargs):
      """
      This function returns raw binary data
      :param bucket:
      :param object_key:
      :return:
      """
      bucket = kwargs.get("Bucket", None)
      object_key = kwargs.get("Key", None)
      data = self.s3_client.get_object(Bucket=bucket, Key=object_key)
      contents = data['Body'].read()
      return contents


  def getObjectToFile(self,bucket, object_key):
      """
      Download an object from S3 to a file.
      :param object_key:
      :return:
      """
      # Create local directory structure
      #local_file_path = "/" + object_key
      filename = os.path.split(object_key)[-1]
      local_file_path = "/var/log/dss/" + filename
      directory = os.path.dirname(local_file_path)
      if not os.path.exists(directory):
          os.makedirs(directory)
      self.s3_client.download_file(Bucket=bucket, Key=object_key, Filename=local_file_path)

  def deleteObject(self, bucket=None,prefix=None):
      """
      Remove a object based on object key.
      :param bucket: Bucket name required
      :param prefix: prefix required
      :return:
      delete_object(Bucket='string',
      Key='string',
      MFA='string',
      VersionId='string',
      RequestPayer='requester',
      BypassGovernanceRetention=True|False,
      ExpectedBucketOwner='string')
      Return:
      {
      'DeleteMarker': True|False,
      'VersionId': 'string',
      'RequestCharged': 'requester'
      }
      """
      if bucket and prefix:
          try:
              response = self.s3_client.delete_object(Bucket=bucket, Key=prefix)
              #print("Response:{}".format(respose))
              """ 
              {'ResponseMetadata': {'RequestId': '16646EFF2BD0F3CB',
               'HostId': '', 'HTTPStatusCode': 204, 
               'HTTPHeaders': {'accept-ranges': 'bytes', 'content-security-policy': 'block-all-mixed-content',
               'server': 'MinIO', 'vary': 'Origin', 'x-amz-request-id': '16646EFF2BD0F3CB', 'x-xss-protection': '1;
               mode=block', 'date': 'Wed, 17 Feb 2021 04:36:48 GMT'}, 'RetryAttempts': 0}}
              """
              if "DeleteMarker" in response:
                  if response["DeleteMarker"]:
                      return True
                  else:
                      print("ERROR: {} not deleted ".format(prefix))
              else:
                  return True   # Require to handle properly.
          except Exception as e:
              print("EXCEPTION: {}".format(e))

      return False





if __name__ =="__main__":

    config = { "name": "dss",
            "credentials": { "endpoint": "http://msl-ssg-vm21-tcp-0:9000", "access_key": "minio", "secret_key": "minio123"}
    }
    start_time = datetime.now()
    s3=  S3(config)
    print("INFO: DSS Client Connection Time: {}".format((datetime.now() - start_time).seconds))
    data = s3.getObject(Bucket="dss0", Key="192.168.200.200/mnt/nfs_share/10gb-B/1MB_0018.dat")
    print(type(data))
    if s3.getObjectToFile("dss0", "192.168.200.200/mnt/nfs_share/10gb-B/1MB_0018.dat") is not None:
        print("ERROR: In downloading object to file ")

    for objects in s3.listObjects(bucket="dss0",prefix="192.168.200.200/mnt/nfs_share/10gb-B"):

        print("Total Objects Listed: {}".format(len(objects)))
        print(objects[0], objects[-1])









