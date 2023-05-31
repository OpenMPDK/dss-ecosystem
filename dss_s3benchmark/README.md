# Introduction
s3-benchmark is a performance testing tool provided by Wasabi for performing S3 operations (PUT, GET, and DELETE) for objects. Besides the bucket configuration, the object size and number of threads varied be given for different tests.

The testing tool is loosely based on the Nasuni (http://www6.nasuni.com/rs/nasuni/images/Nasuni-2015-State-of-Cloud-Storage-Report.pdf) performance benchmarking methodologies used to test the performance of different cloud storage providers

# Prerequisites
To leverage this tool, the following prerequisites apply:
*	Git development environment
*	Ubuntu Linux shell programming skills
*	Access to a Go 1.7 development system (only if the OS is not Ubuntu Linux 16.04)
*	Access to the appropriate AWS EC2 (or equivalent) compute resource (optimal performance is realized using m4.10xlarge EC2 Ubuntu with 10 GB ENA)
*	Dependency: Download github.com/openMPDK/dss-sdk repo and compile

# Building the Program
Obtain a local copy of the repository using the following git command with any directory that is convenient:

```
git clone --recursive git@MSL-DC-GITLAB.SSI.SAMSUNG.COM:ssd/nkv-s3benchmark.git s3-bench
```

You should see the following files in the s3-benchmark directory.
LICENSE	README.md		s3-benchmark.go	s3-benchmark.ubuntu

If the test is being run on Ubuntu version 16.04 LTS (the current long term release), the binary
executable s3-benchmark.ubuntu will run the benchmark testing without having to build the executable. 

Otherwise, to build the s3-benchmark executable, you must issue this following command:

*	cd /root/s3-benchmark
*	export GO111MODULE=off
*	export GOPATH=/root/gopath
*	export PATH=$PATH:$GOPATH/bin:/root/go/bin
*       export  CGO_LDFLAGS="-L/root/som/dss-ecosystem/dss_s3benchmark/ -lrdmacm -libverbs -ldss"
*       export  CGO_CFLAGS="-std=gnu99 -I/root/som/dss-ecosystem/dss_client/include/"	

*	go get -u github.com/aws/aws-sdk-go/aws/...
*	go get -u github.com/aws/aws-sdk-go/service/...
*	go get -u code.cloudfoundry.org/bytefmt

*	go build s3-benchmark.go
 
# Command Line Arguments
Below are the command line arguments to the program (which can be displayed using -help):

```
./s3-benchmark --help
Wasabi benchmark program v2.0
Usage of myflag:
  -a string
        Access key
  -b string
        Bucket for testing (default "wasabi-benchmark-bucket")
  -c int
        Number of object per thread written earlier
  -d int
        Duration of each test in seconds (default 60)
  -l int
        Number of times to repeat test (default 1)
  -n int
        Number of IOS per thread to run
  -o int
        Type of op, 1 = put, 2 = get, 3 = del
  -p string
        Key prefix to be added during key generation (default "s3-bench-minio")
  -r string
        Region for testing (default "us-east-1")
  -s string
        Secret key
  -t int
        Number of threads to run (default 1)
  -u string
        URL for host with method prefix (default "http://s3.wasabisys.com")
  -z string
        Size of objects in bytes with postfix K, M, and G (default "1M")

```        

# Example Benchmark
Below is an example run of the benchmark for 10 threads with the default 1MB object size.  The benchmark reports
for each operation PUT, GET and DELETE the results in terms of data speed and operations per second.  The program
writes all results to the log file benchmark.log.

```
ubuntu:~/s3-benchmark$ ./s3-benchmark.ubuntu -a MY-ACCESS-KEY -b jeff-s3-benchmark -s MY-SECRET-KEY -t 10 
Wasabi benchmark program v2.0
Parameters: url=http://s3.wasabisys.com, bucket=jeff-s3-benchmark, duration=60, threads=10, loops=1, size=1M
Loop 1: PUT time 60.1 secs, objects = 5484, speed = 91.3MB/sec, 91.3 operations/sec.
Loop 1: GET time 60.1 secs, objects = 5483, speed = 91.3MB/sec, 91.3 operations/sec.
Loop 1: DELETE time 1.9 secs, 2923.4 deletes/sec.
Benchmark completed.

./s3-benchmark -a minio -b som4 -s minio123 -u http://10.1.51.21:9000 -t 100 -z 10M -n 100 -o 1
Wasabi benchmark program v2.0
Parameters: url=http://10.1.51.21:9000, bucket=som4, region=us-east-1, duration=60, threads=100, num_ios=100, op_type=1, loops=1, size=10M
2021/01/05 18:22:52 WARNING: createBucket som4 error, ignoring BucketAlreadyOwnedByYou: Your previous request to create the named bucket succeeded and you already own it.
        status code: 409, request id: 1657835058733E40, host id:
Loop 1: PUT time 38.9 secs, objects = 10000, speed = 2.5GB/sec, 257.1 operations/sec. Slowdowns = 0

./s3-benchmark -a minio -b som4 -s minio123 -u http://10.1.51.21:9000 -t 100 -z 10M -n 100 -o 2
Wasabi benchmark program v2.0
Parameters: url=http://10.1.51.21:9000, bucket=som4, region=us-east-1, duration=60, threads=100, num_ios=100, op_type=2, loops=1, size=10M
2021/01/05 18:23:39 WARNING: createBucket som4 error, ignoring BucketAlreadyOwnedByYou: Your previous request to create the named bucket succeeded and you already own it.
        status code: 409, request id: 1657835B38D61A94, host id:
Loop 1: GET time 14.9 secs, objects = 10000, speed = 6.6GB/sec, 672.1 operations/sec. Slowdowns = 0

./s3-benchmark -a minio -b som4 -s minio123 -u http://10.1.51.21:9000 -t 100 -z 10M -n 100 -o 3
Wasabi benchmark program v2.0
Parameters: url=http://10.1.51.21:9000, bucket=som4, region=us-east-1, duration=60, threads=100, num_ios=100, op_type=3, loops=1, size=10M
2021/01/05 18:24:04 WARNING: createBucket som4 error, ignoring BucketAlreadyOwnedByYou: Your previous request to create the named bucket succeeded and you already own it.
        status code: 409, request id: 16578360FD53602A, host id:
Loop 1: DELETE time 4.3 secs, 2342.4 deletes/sec. Slowdowns = 0

//Prefix based run
root@msl-ssg-si04 s3-bench]./s3-benchmark -a minio -b som10 -s minio123 -u http://10.1.51.21:9000 -t 100 -z 1M -n 100 -o 1 -p samsung-s3-bench
Wasabi benchmark program v2.0
Parameters: url=http://10.1.51.21:9000, bucket=som10, region=us-east-1, duration=60, threads=100, num_ios=100, op_type=1, loops=1, size=1M
2021/02/17 17:08:12 WARNING: createBucket som10 error, ignoring BucketAlreadyOwnedByYou: Your previous request to create the named bucket succeeded and you already own it.
        status code: 409, request id: 1664B231AFA7F9A4, host id:
Loop 1: PUT time 4.6 secs, objects = 10000, speed = 2.1GB/sec, 2157.8 operations/sec. Slowdowns = 0

//Time based read

root@msl-ssg-si04 s3-bench]./s3-benchmark -a minio -b som10 -s minio123 -u http://10.1.51.21:9000 -t 100 -z 1M -d 30 -c 100 -o 2 -p samsung-s3-bench
Wasabi benchmark program v2.0
Parameters: url=http://10.1.51.21:9000, bucket=som10, region=us-east-1, duration=30, threads=100, num_ios=0, op_type=2, loops=1, size=1M
2021/02/17 17:06:55 WARNING: createBucket som10 error, ignoring BucketAlreadyOwnedByYou: Your previous request to create the named bucket succeeded and you already own it.
        status code: 409, request id: 1664B21FB5A67930, host id:
Loop 1: GET time 30.0 secs, objects = 201106, speed = 6.5GB/sec, 6700.2 operations/sec. Slowdowns = 0

//Gen2 RDD way
./s3-benchmark -a minio -b rddtest207 -s minio123 -u http://208.0.0.33:9000 -t 100 -z 1M -c 1000 -d 20 -o 2 -p dss-gen2-test-207 -rdd 1 -rdd_ips 207.0.0.30,207.0.0.31,207.0.0.32,207.0.0.33 -rdd_port 1234 -instance_id 3

```

# Note
Your performance testing benchmark results may vary most often because of limitations of your network connection to the cloud storage provider.  Wasabi performance claims are tested under conditions that remove any latency (which can be shown using the ping command) and bandwidth bottlenecks that restrict how fast data can be moved.  For more information,
contact Wasabi technical support (support@wasabi.com).
