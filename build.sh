#!/bin/bash

rm -rf ./build
mkdir ./build
cd ./build
CXX=/opt/gcc/bin/g++ cmake ../
make -j
cd ../
