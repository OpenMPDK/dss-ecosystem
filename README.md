Build DSS client library:

Platform requirements: CentOS 7.x, gcc that fully supports C++11, cmake 3.7 or above
					   Python 3.8 or above
Library dependencies: aws-cpp-sdk with core, S3 services enabled

There is a build.sh that wraps CMakeLists.txt that automatically produces a C++ libdss.so file
and a dss.cpython-XXXXXX.so which provides python interface under ./build directory.

To validate the python extension, start python shell and run
>>> import dss
and you should not see any errors


Building gcc:
yum install dnf
dnf install libmpc-devel mpfr-devel gmp-devel zlib-devel*
git clone https://gcc.gnu.org/git/gcc.git
git checkout releases/gcc-9.3.0
cd gcc
./configure --with-system-zlib --disable-multilib --enable-languages=c,c++ --prefix=/opt/gcc
make -j;make install

create /etc/ld.so.conf.d/stdlibc_cxx.conf with the lines:
/opt/gcc/lib
/opt/gcc/lib64


Install Anaconda:
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +X Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh


Install cmake:
pip3 install cmake


Build aws-cpp-sdk:
yum install libcurl-devel openssl-devel libuuid-devel pulseaudio-libs-devel
git clone https://github.com/aws/aws-sdk-cpp.git
git checkout 1.8.99
mkdir build
cd build
/opt/gcc/bin/g++ cmake ../ -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3"

