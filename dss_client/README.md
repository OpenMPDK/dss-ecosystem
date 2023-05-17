# Client Library

## Platform Requirements

- CentOS 7.x
- gcc that fully supports C++11, cmake 3.13 or above
- Python 3.8 (3.6 and 3.7 are probably ok but not verfied)
- Library dependencies: aws-cpp-sdk v1.9 with core, S3 services enabled, pybind11, python3-config
- Dependency: github.com/openMPDK/dss-sdk repo is required and to be compiled

## Build

Simply execute `./scripts/build.sh`

The build produces `libdss.so` and `dss.cpython-XXXXXX.so` which provide the python interface
under the `./build` directory. Additionally, a release tarball `dss_client-#GITHASH#.tgz` file is generated for use with automated deployment.

### Build Setup Enviroment

The easiest way to setup the build env is to use Anaconda to install gcc, cmake, etc. , if Anacoda is not available, one can manully build gcc and cmake.

#### Anaconda

- Install Anaconda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
```

- Install pybind11 and aws-cpp-sdk

```bash
Install pybind11
sudo pip3 install pybind11
```

- Use aws-cpp-sdk rpm, if not available build aws-cpp-sdk from the source

```bash
Build aws-cpp-sdk:
yum install libcurl-devel openssl-devel libuuid-devel pulseaudio-libs-devel

GIT_CHECKOUT_TAG="1.9.343-elbencho-tag"
AWS_DIR="aws-git-1.9"
INSTALL_DIR="/usr/local/"

git clone --recursive https://github.com/breuner/aws-sdk-cpp.git ${AWS_DIR}
cd ${AWS_DIR}
git checkout ${GIT_CHECKOUT_TAG}


Apply the following patch to build dynamic libraries

--- aws-git-1.9/CMakeLists.txt  2022-12-06 13:37:22.790891000 -0800
+++ aws-git-1.9.new/CMakeLists.txt      2022-12-06 13:56:46.023876000 -0800
@@ -204,6 +204,7 @@
         set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_PREV})
     endif()
 else()
+    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/crt/aws-crt-cpp/crt/aws-c-common/cmake")
     include(AwsFindPackage)
     set(IN_SOURCE_BUILD OFF)
 endif()

source /opt/rh/devtoolset-11/enable
pushd crt/aws-crt-cpp
cmake3 . -DBUILD_SHARED_LIBS=ON -DCPP_STANDARD=17 -DAUTORUN_UNIT_TESTS=OFF -DENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DBYO_CRYPTO=ON -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} && sudo make -j $(nproc) install
popd
cmake3 . -DBUILD_ONLY="s3;transfer" -DBUILD_SHARED_LIBS=ON -DCPP_STANDARD=17 -DAUTORUN_UNIT_TESTS=OFF -DENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DBYO_CRYPTO=ON -DBUILD_DEPS=OFF -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} && sudo make -j $(nproc) install
popd

```

#### Manual Build

- Install dependencies

```bash
sudo yum install -y git libmpc-devel mpfr-devel gmp-devel zlib-devel gcc gcc-c++ openssl-devel libcurl-devel libuuid-devel pulseaudio-libs-devel python3 python3-pip python3-devel cmake3
```

- Install devtoolset-11
```bash
sudo yum install -y centos-release-scl-rh-2-3.el7.centos.noarch.rpm
sudo yum install -y devtoolset-11
```

- Build client library

```bash

sudo yum install -y rdma-core-devel

cd ~/
git -c http.sslVerify=false clone https://msl-dc-gitlab.ssi.samsung.com/ssd/dss_client.git
cd dss_client
./scripts/build.sh
```
Make sure the libraries (libdss.so) in the build directory are present in the /usr/local/lib64 either as a file or a symbolic link and the ldconf file is updated to recognize that directory for linking

- Validate the python extension, by starting python shell and run the following code, you should be able see a list of classes offered by client library

```python
 >>> import dss
 >>> print(help(dss))
```

## Test

Use test/example.py to verify and test the build also it can serves as
an example for the client APIs (refer to client lib API documentation). It uploads, lists then downloads example.py
as a test file against S3 store. In this case, the S3 store can be a stock minio
or DSS cluster.

Note: Before the run, you need to config S3 endpoint in the header section of the file,

```ini
access_key = "minioadmin"
access_secret = "minioadmin"
discover_endpoint = 'http://127.0.0.1:9001'
```

If you are working with a new DSS clusters, you need to prime one of the cluster
to contain a cluster config file for client library to discover cluster topology. In order
to do it, pick a cluster to create a bucket named "dss" and upload a file named "conf.json"
to it. See conf.json example in the source tree.

## Debug

To enable aws-cpp-sdk logging, set environment variable DSS_AWS_LOG to the range between 0 and 6.

```c++
enum class LogLevel : int
{
    Off = 0,
    Fatal = 1,
    Error = 2,
    Warn = 3,
    Info = 4,
    Debug = 5,
    Trace = 6
};
```

In case you need to run client lib in multiple processes, you can direct each client
process to have it's own aws log file name using environment variable DSS_AWS_LOG_FILENAME.

To enable dss logging, set env var DSS_DEBUG to the source file names seperated by
comma.

```DSS_DEBUG=/path/to/src/dss_client.cpp python3 test/test.py```

To feed conf.json from local filesystem, set environment variable DSS_CONFIG_FILE to the path
pointing to the config file

```DSS_CONFIG_FILE=/path/to/clientlib/dss_client/conf.json python3 test/example.py```


##API Reference

- createClient (url, username, password, options)

Creates a client object with URL and the credentials

**URL** is the S3 minio instance of the format http://\<ip\>:\<port\>

**username** and **password** are the minio instance credentials

Returns: A client object to use for get/put/del objects

The following APIs are the functions of the client object instance created with createClient()

- deleteObject(key)

Deletes the object with the name *key*

Returns: 0 on success, -1 on failure

- getObject(key, file_name)

Download the object with the name *key* to the file name

Returns: 0 on success, -1 on failure

- getObjectBuffer(key, buffer)

Get the object into a bytearray buffer. Allocation and release of buffer is the caller's responsibility

Returns: Actual data length in the buffer, -1 on failure

-  getObjectNumpyBuffer(key, numpy_buffer)

Get the object into a numpy buffer. Allocation and release of buffer is the caller's responsibility

Returns: Actual data length in the buffer, -1 on failure

- getObjects(prefix, delimiter, common_prefix, limit)

Returns list of objects matching with the prefix
Need to call this in a recursive manner until the end of iterator
```
    objects = list()
    try:
        obj_iter = self.client.getObjects(prefix)
        while True:
            try:
                for file in obj_iter:
                    objects.append(file)
            except dss.NoIterator:
                break
    except Exception as e:
        raise e

    return objects
```
- putObject(key, file_name)

Upload the object with the name *key* to the file name

Returns: 0 on success, -1 on failure
