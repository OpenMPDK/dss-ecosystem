#!/usr/bin/python


import os,sys
import time
import random
import argparse
import platform
from multiprocessing import Process,Queue
from utils.utility import exception, exec_cmd
from utils.config import Config, commandLineArgumentParser, CommandLineArgument
from minio_client import MinioClient
from nfs_cluster import NFSCluster




def check_operation(operation_list=None, operation=None):
  if operation_list:
    if operation not in operation_list:
      print("ERROR: Operation {} not supported!! \n Supported operations {} ".format(operation,operation_list))
      sys.exit()
    else:
      print("DEBUG: Operation {} is supported!!".format(operation)) # Need to remove




def create_bucket(s3_client, bucket=None):
  """

  :param s3_client: s3 client
  :param bucket:
  :param local_nfs_mount_paths:
  :return: None
  # To do
   Support of many buckets <cluster-ip>-<top nfs dir>
  """
  if bucket:
    s3_client.make_bucket(bucket)
  #else:
  #  for cluster_ip in local_nfs_mount_paths:
  #    s3_client.make_bucket(cluster_ip.replace(".","-"))

      # This is for many buckets based on <cluster-ip>-<top level nfs dir>
      #for nfs_share in local_nfs_mount_paths[cluster_ip]:
      #  # Create bucket
      #  mc.make_bucket(cluster_ip + "-{}".format(nfs_share[1:]))

    



def main():

  cli = CommandLineArgument()
  print(cli.operation)
  print(cli.options)
  #operation, params = commandLineArgumentParser()

  params = cli.options
  config_obj = Config(params)
  config = config_obj.get_config()
  #print(config)
  job_count = params.get("thread", 5)
  bucket = params.get("bucket", None)
  #operation = params.get("operation")
  operation = cli.operation
  prefix = params.get("prefix", None)

  # Check if specified operation is supported
  check_operation(config["operations"], operation)

  nfs_cluster_obj =  NFSCluster(config.get("nfs_config",{}))
  nfs_cluster_obj.mount()
  #local_nfs_mount_paths = nfsMount(config.get("nfs_config"))

  # Minio specific
  minio_config = config["s3_storage"]["minio"]
  #print(minio_config)
  mc = MinioClient(minio_config["url"],minio_config["access_key"],minio_config["secret_key"])

  if bucket:
    create_bucket(mc,bucket)

  # Submit Job

  # Test Copy objects
  nfs_shares = nfs_cluster_obj.get_mounts()
  for cluster_ip in nfs_shares:
    if not bucket:
      bucket = cluster_ip.replace(".","-")
      create_bucket(mc, bucket)
    for data in nfs_shares[cluster_ip]:
      mc.put(bucket,data)

  # Test list operation
  for cluster_ip in nfs_shares:
    if not bucket:
      bucket = cluster_ip.replace(".","-")

    for data in nfs_shares[cluster_ip]:
      print("Listing:")
      objects = mc.list(bucket,data,True)
      for obj in objects:
        print(obj.object_name)

    #print("Remove-{}".format(bucket))
    #mc.delete(bucket)




if __name__ == "__main__":
  main()
