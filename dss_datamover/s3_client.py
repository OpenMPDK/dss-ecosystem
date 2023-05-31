#!/usr/bin/python

"""
# The Clear BSD License
#
# Copyright (c) 2023 Samsung Electronics Co., Ltd.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Samsung Electronics Co., Ltd. nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import boto3
from botocore.client import Config
from datetime import datetime


class S3(object):
    def __init__(self, config={}):
        self.s3_client = boto3.client('s3',
                                      endpoint_url=config["endpoint"],
                                      aws_access_key_id=config["minio_access_key"],
                                      aws_secret_access_key=config["minio_secret_key"],
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
        if os.path.exists(source_file_path) and os.path.isfile(source_file_path) and not object_key.startswith("/"):
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
        object_keys = []
        if bucket and prefix:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)
            # print(response)
            for content in response.get("Contents", {}):
                object_keys.append(content["Key"])
                # yield content["Key"]

        return object_keys

    def getObject(self, object_key):
        # Create local directory structure
        local_file_path = "/" + object_key
        directory = os.path.dirname(local_file_path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.s3_client.download_file(object_key, local_file_path)

    def deleteObject(self, bucket=None, prefix=None):
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
                # print("KKKKKKKKKKKKKK:{}".format(ret))
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
                    return True  # Require to handle properly.
            except Exception as e:
                print("EXCEPTION: {}".format(e))

        return False


if __name__ == "__main__":

    config = {"endpoint": "http://202.0.0.103:9000", "minio_access_key": "minio", "minio_secret_key": "minio123"}
    start_time = datetime.now()
    s3 = S3(config)
    print("INFO: DSS Client Connection Time: {}".format((datetime.now() - start_time).seconds))
    # Example of uploading files
    print("Uploading files")
    directory = "/deer/deer1/"
    uploaded_file_count = 0
    now = datetime.now()

    for file in os.listdir(directory):
        file_path = directory + file
        if s3.putObject("bucket", file_path):
            uploaded_file_count += 1
    print("Uploaded Files:{} , Time-{}".format(uploaded_file_count, (datetime.now() - now).seconds))

    # Example of listing buckets.
    print("All Buckets: {}".format(s3.list_buckets()))

    prefix = "deer/"
    # List object keys
    print("List Object Keys: {}".format(s3.listObjects("bucket", prefix)))

    # Delete object

    deleted_object_count = 0
    for object_key in s3.listObjects("bucket", prefix):
        # print(object_key)
        if s3.deleteObject("bucket", object_key):
            deleted_object_count += 1

    print("Deleted Objects Count: {} for prefix-\"{}\"".format(deleted_object_count, prefix))
