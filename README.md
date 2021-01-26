# nkv-datamover


Execution command:
python3 tess_copy.py -op PUT -c 10.1.51.2

Master Application Node: msl-dc-client3

NFS Cluster: msl-dc-client7

Client Nodes:
msl-dpe-perf35:10.1.51.91
msl-dpe-perf36:10.1.51.132
msl-dpe-perf37:10.1.51.107

Dependency:
Install following packages on the client nodes before launching client application to those nodes.
python3 -m pip install paramiko
python3 -m pip install boto3
python3 -m pip install pyminio
python3 -m pip install ntplib

