# Client Library

## Platform Requirements

- CentOS 7.x
- gcc that fully supports C++11, cmake 3.13 or above
- Python 3.8 (3.6 and 3.7 are probably ok but not verfied)
- Library dependencies: aws-cpp-sdk with core, S3 services enabled, pybind11, python3-config

## Build

In source code tree, there is a build.sh can be used as reference to build Client lib which
consists of libdss.so and dss.cpython-XXXXXX.so which provides python interface
under ./build directory.

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
git clone https://github.com/aws/aws-sdk-cpp.git
git checkout 1.8.99
mkdir build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=/opt/gcc/bin/g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3"
make
sudo make install
```

#### Manual Build

- Install dependencies

```bash
sudo yum install -y git libmpc-devel mpfr-devel gmp-devel zlib-devel gcc gcc-c++ openssl-devel libcurl-devel libuuid-devel pulseaudio-libs-devel python3 python3-pip python3-devel
```

- Build GCC 9.3.0

```bash
git clone https://gcc.gnu.org/git/gcc.git
cd gcc
git checkout releases/gcc-9.3.0
./configure --with-system-zlib --disable-multilib --enable-languages=c,c++ --prefix=/opt/gcc
make -j$(nproc)
sudo make install
sudo sh -c 'echo "/opt/gcc/lib" > /etc/ld.so.conf.d/stdlibc_cxx.conf'
sudo sh -c 'echo "/opt/gcc/lib64" >> /etc/ld.so.conf.d/stdlibc_cxx.conf'
sudo ldconfig
```

- Build cmake 3.18.4

```bash
cd ~/
git clone https://gitlab.kitware.com/cmake/cmake.git
cd cmake
git checkout v3.18.4
./bootstrap --prefix=/opt/cmake
gmake
sudo gmake install
```

- Add cmake to path

```bash
sed -i 's|^PATH=.*|&:/opt/cmake/bin|g' ~/.bash_profile
source ~/.bash_profile
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
git clone https://github.com/aws/aws-sdk-cpp.git
git checkout 1.8.99
mkdir build
cd build
cmake ../ -DCMAKE_CXX_COMPILER=/opt/gcc/bin/g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3"
make
sudo make install
```

- Build client library

```bash
cd ~/
git -c http.sslVerify=false clone https://msl-dc-gitlab.ssi.samsung.com/ssd/dss_client.git
cd dss_client
./build.sh
```

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
