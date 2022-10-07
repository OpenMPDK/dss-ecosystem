# SyntheticDataGenerator

A custom tool to generate synthetic data from a small dataset. The tool copy the data from source to destination.
The source of data can be from a posix based file system or from S3 Object storage. Similarly, the destination
 either would be file system or S3 Object storage. A replication factor and the destination directories
 determines the overall size of the copied data in the destination. The tool preserve the
 source directory structure in the destination.

 Let say, source data size is = 1GiB, replication factor= 2, destination directories = 2
 Then each file of the source would be copied twice under a directory. As, the above example
 has 2 directories, then there would 2x2 = 4 copies of a source file.

 ```text
Replication Factor: 2
Source: dir1
dir1/file[1,10]
dir1/dir11/file[1,10]
dir1/dir12/file[1,10]
Destination: data1
data1/dir1/file[1,10]_[0,1]
dir1/dir11/file[1,10]_[0,1]
dir1/dir12/file[1,10]_[0,1]
```

## Setup and Installation

Install the required libraries by running the following command in the node you are working on.

```bash
python3 -m pip install -r requirements.txt
```

To work with dss_client lib, one require to install that library separately on the node.

## Configuration

The user requires to set the source and destination path of data in the configuration file.
The workers, determine how many parallel operations are to be performed for listing and copy.
User should set source and destination data type.
Supported client_lib : "dss_client", "boto3"

```json
{
  "workers": 5,
  "replication": {
    "factor": 3,
    "max_size": 1024
  },
  "source": {
    "type": "s3",
    "storage": {
      "fs": {
        "name": "nfs",
        "paths": ["<source data1>", "<source data2>", "<source data3>"]
      },
      "s3": {
        "name": "dss",
        "bucket": "dss0",
        "credentials": {
          "endpoint": "http://<ip-address>:<port>",
          "access_key": "minio",
          "secret_key": "minio123" },
        "client_lib": {
          "name": "dss_client",
          "dss_client": {"max_connections":25, "http_request_timeout_ms":0, "connect_timeout_ms":1000,
                         "request_timeout_ms": 10000, "enable_tcp_keep_alive":true,
                         "tcp_keep_alive_interval_ms":  1000, "object_keys_per_page_count":  1000}
        },
        "prefixes": ["<source prefix1>/", "<source prefix2>/"]
      }
    }
  },
  "destination": {
    "type": "fs",
    "storage": {
      "fs": {
        "name": "nfs",
        "paths": ["<destination path1>", "<destination path2>"]
      },
      "s3": {
        "name": "dss",
        "bucket": "dss0",
        "credentials": {
          "endpoint": "http://<ip address>:<port>",
          "access_key": "minio",
          "secret_key": "minio123" },
        "client_lib": {
          "name": "dss_client",
          "dss_client": {"max_connections":25, "http_request_timeout_ms":0, "connect_timeout_ms":1000,
                         "request_timeout_ms": 10000, "enable_tcp_keep_alive":true,
                         "tcp_keep_alive_interval_ms":  1000, "object_keys_per_page_count":  1000}
        },
        "prefixes": ["<destination prefix>1/", "<destination prefix2>/"]
      }
    }
  },
  "logging": { "path": "/var/log/dss", "level": "INFO"}
}
```

## Logging

Set the logging path in the following section.

```json
"logging": { "path": <log_path>, "level": "INFO"}
```

## Execution

Once all the configuration is done, run the tool as below.

```bash
# Use configuration file from default location.
python3 generate_data.py

# Use configuration file from custom location
python3 generate_data.py -cfg <configuration fle>

# To run with dss_cleint lib use thw following command
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 generate_data.py '
```

## Check all running process

```bash
[user@node]$ ps -e | grep SDG
 1410 pts/0    00:00:00 SDG_logger
 1421 pts/0    00:00:17 SDG_worker_0
 1422 pts/0    00:00:17 SDG_worker_1
 1423 pts/0    00:00:17 SDG_worker_2
 1424 pts/0    00:00:17 SDG_worker_3
 1425 pts/0    00:00:17 SDG_worker_4
 
 # Kill running process
 [user@node]$ sudo pkill SDG*
```
