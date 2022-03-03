#!/bin/bash

rm -rf ./build
mkdir ./build
cd ./build
CXX=/opt/gcc/bin/g++ cmake -DWITH_RELEASE=1 ../
make -j
make install
cd ../
