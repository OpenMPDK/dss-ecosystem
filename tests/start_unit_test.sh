#!/bin/bash
export PYTHONPATH="/usr/dss/nkv-datamover"
echo "Start DataMover UnitTest"
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3  -m unittest unit_test.DataMover'
