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
  def __init__(self, **kwargs):
      self.storage_name = kwargs["storage_name"]
      self.credentials = kwargs["credentials"]
      self.logger = kwargs["logger"]
      self.get_s3_client()

  def get_s3_client(self):
      if self.storage_name.lower() == "aws":
          self.s3_client = boto3.client('s3',
                        aws_access_key_id=self.credentials["access_key"],
                        aws_secret_access_key=self.credentials["secret_key"],
                        config=Config(signature_version='s3v4'),
                        region_name='us-east-1')
      elif self.storage_name.lower() == "dss":
          self.s3_client = boto3.client('s3',
                                        endpoint_url=self.credentials["endpoint"],
                                        aws_access_key_id=self.credentials["access_key"],
                                        aws_secret_access_key=self.credentials["secret_key"])
      else:
          self.logger.error("Wrong object storage name!")

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
              self.logger.error("{}".format(e))
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
          self.logger.error("Requires bucket name!")
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
          self.logger.excep("{}".format(e))

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
              self.logger.error("listObjects:{}".format(e))


  def getObject(self, **kwargs):
      """
      This function returns raw binary data
      :param bucket:
      :param object_key:
      :return:
      """
      bucket = kwargs.get("bucket", None)
      object_key = kwargs.get("key", None)
      data = self.s3_client.get_object(Bucket=bucket, Key=object_key)
      contents = data['Body'].read()
      return contents


  def getObjectToFile(self,**kwargs):
      """
      Download an object from S3 to a file.
      :param bucket: bucket name to read from
      :param key: object key to download object from the bucket
      :param dest_file_path: store the file in the mentioned path, by default "/dev/null"
      :return:
      """
      bucket= kwargs["bucket"]
      object_key=kwargs["key"]
      dest_file_path=kwargs["dest_file_path"]
      if dest_file_path == "/dev/null":
          local_file_path = dest_file_path
      else:
          # Create local directory structure
          local_file_path = dest_file_path + "/" + object_key
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
                      self.logger.error("{} not deleted ".format(prefix))
              else:
                  return True   # Require to handle properly.
          except Exception as e:
              self.logger.error("{}".format(e))

      return False










