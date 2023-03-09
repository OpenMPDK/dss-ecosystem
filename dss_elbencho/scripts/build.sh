make clean
source /opt/rh/devtoolset-11/enable 
make S3_SUPPORT=1 AWS_INCLUDE_DIR=/usr/local/include/ AWS_LIB_DIR=/usr/local/lib64  DSS_INCLUDE_DIR=../dss_client/include BUILD-VERBOSE=1 -j 2

