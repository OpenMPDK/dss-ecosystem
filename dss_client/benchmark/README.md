# Introduction

This benchmark tool is a performance testing tool similar to S3 benchmark. It uses Samsung's proprietary library "dss"
to distribute the files or objects across the MINIO cluster. It uses the files in the directory rather than object in memory.
The tool tests the performance of PUT/GET/DEL/LIST calls.

## Prerequisites

* aws-sdk-cpp library - Can be downloaded from wget <https://codeload.github.com/aws/aws-sdk-cpp/tar.gz/1.8.99>
and build with gcc 9.3
* dss library - Samsung's proprietary library for data distribution across single or multi cluster setup

## Usage

The tool can be used to generate files with a set prefix.
For a given prefix, the thread count and the number of IOs
per thread, the tool generates files of the format \<prefix\>-object-\<thread_ID\>-\<IO_num_per_thread\>

It is the responsibility of the user to prepare data (```-o 8```) and cleanup data (```-o 9```) before performing
PUT/GET/DEL calls.

The help usage for this tool is

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -h
usage: benchmark.py [-h] -a ACCESS_KEY -s SECRET_KEY -u ENDPOINT_URL
                    [-l TOTAL_LOOPS] -n NUM_IOS [-o {0,1,2,3,4,5,8,9}]
                    [-t THR_CNT] [-z OBJECT_SIZE] [-p KEY_PREFIX]
                    [-x DATA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -a ACCESS_KEY, --access-key ACCESS_KEY
                        Access Key of the Minio server
  -s SECRET_KEY, --secret-key SECRET_KEY
                        Secret Key of the Minio server
  -u ENDPOINT_URL, --endpoint-url ENDPOINT_URL
                        Endpoint URL of MINIO server
  -l TOTAL_LOOPS, --loops TOTAL_LOOPS
                        Number of loops to run (default: 1)
  -n NUM_IOS, --num_ios NUM_IOS
                        Number of IOs to do (default: 1)
  -o {0,1,2,3,4,5,8,9}, --op_type {0,1,2,3,4,5,8,9}
                        Type of IO (1 - PUT, 2 - GET, 3 - DEL, 4 - LIST, 5 - GET WITH INTEGRITY CHECK, 8 -
                        PREPARE DATA FOR PUT, 9 - CLEANUP, 0 - PUT/GET/DEL)
  -t THR_CNT, --num_threads THR_CNT
                        Number of threads to start (default: 1)
  -z OBJECT_SIZE, --object-size OBJECT_SIZE
                        size of object in KB (default:1024)
  -p KEY_PREFIX, --key-prefix KEY_PREFIX
                        Key prefix for the object name
  -x DATA_DIR, --data-dir DATA_DIR
                        Data directory to read from/write to
```

## Examples

* To run the performance test for PUT/GET/DEL calls for 1000 files of 1MB size per thread and 20 thread, run the command

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -u http://202.0.0.1:9000 -a minio -s minio123 -n 1000 -o 0
-t 20 -z 1024 -p <myprefix> -x <mydir>
```

The above command prepares the files, perform PUT/GET/DEL on all the files and prints the throughput details on the console.

* To create 1000 files of 1MB size per thread and 20 thread, run the command

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -u http://202.0.0.1:9000 -a minio -s minio123 -n 1000 -o 8
-t 20 -z 1024 -p <myprefix> -x <mydir>
```

* To run the performance test for only PUT calls for the above configuration, run

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -u http://202.0.0.1:9000 -a minio -s minio123 -n 1000 -o 1
-t 20 -z 1024 -p <myprefix> -x <mydir>
```

***Running performance test for  PUT requires the files to be present before. Use "-o 8" option to create files prior***

* To run the performance test for only PUT calls in a particular custom directory of the destination folder for the above
configuration, run

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -u http://202.0.0.1:9000 -a minio -s minio123 -n 1000 -o 1
-t 20 -z 1024 -p <custom_dir>/<myprefix> -x <mydir>
```

* To run the performance test for only GET calls for the above configuration, run

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -u http://202.0.0.1:9000 -a minio -s minio123 -n 1000 -o 2
-t 20 -z 1024 -p <myprefix> -x <mydir>
```

* To run the performance test for only DEL calls for the above configuration, run

```bash
[ansible@msl-dpe-da1 benchmark]# python3 benchmark.py -u http://202.0.0.1:9000 -a minio -s minio123 -n 1000 -o 3
-t 20 -z 1024 -p <myprefix> -x <mydir>
```

## Debugging

* Check the dss_benchmark.log file for any errors happened during runtime
