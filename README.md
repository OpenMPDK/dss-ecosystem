# nkv-datamover

# Dependency:
Install following packages on the client nodes before launching client application to those nodes.
```
python3 -m pip install paramiko
python3 -m pip install boto3
python3 -m pip install pyminio
python3 -m pip install ntplib

Use isntall.sh on each master/client node.
OR
pip3 install requirements.txt
```

# Execution command:
```
python3 master_application.py PUT -c 10.1.51.2
python3 master_application.py LIST -c 10.1.51.2
python3 master_application.py DEL -c 10.1.51.2
python3 master_application.py DEL -c 10.1.51.2 --prefix bird/
python3 master_application.py DEL -c 10.1.51.2 --dest_path <Destination Path>

Dry Run:
- Read files from NFS shares, but skip the upload operation. Show RAW NFS read performance
python3 master_application.py PUT -c 10.1.51.2 --dryrun 
- It performa every steps involved in DELETE operation except actual DELETE from S3 storage.
python3 master_application.py DEL -c 10.1.51.2 --dryrun
- It performs every steps involved in GET operation except actual S3 GET operation
python3 master_application.py DEL -c 10.1.51.2 --dest_path <"Destination File Path"> --dryrun 
NFS Cluster: 10.1.51.2
```
# TESS Master Node and Client Nodes
## Master Node
   The master node is the node where, the master_application starts. 
   It does indexing for PUT operation and distribute the file index to the ClientNodes in
   a "RoundRobin" fashion. The master_Application uses multiple workers to perform indexing.
   The MasterApplication also consist three monitors such as IndexMonitor, StatusPoller, StatusProgressCalculation.
   IndexMonitor: Distribute the file or object_key indexes  to the client nodes through message queue.
   StatusPoller: Read status messages from each clients and add them to a status queue.
   StatusProgress: Read all status data from status queue and calculate the progress.
                   Display the result.
   The file upload is performed by the ClientApplication running in each TESS client node.
### MasterApplication configuration
Update master section in the config.json
```
 "master": {
    "ip_address": "202.0.0.135", # Make sure master IP address is sepcified correctly
    "workers": <Specify no of workers>,
    "max_index_size": <Maximum no of indexes to be sent through a Message>,
    "size": <Overall Size of each set of message, say 1GB. Thats mean the message to be sent with 
    x number of file name of size 1GB>
  },
  
  Default is file count based indexing. 
  #TODO:
  - Incorporate size based indexing too. Support size based indexing if specified.
``` 
## Client Nodes:
   The ClientApplication running in each client node performs the actual operation PUT/DEL/GET.
   Further, the received indexes are in divided in small tasks and added to TaskQ. Multiple workers 
   running on that client node, process each task independently and update status in a shared queue.
   
### How to configure ClientApplication
Specify TESS client nodes, ip address or DNS name
```
"tess_clients_ip":["10.1.51.91", "10.1.51.132", "10.1.51.107"],
```
Update client section in config.json
``` 
 "client": {
    "workers": <Specifiy Workers>,
    "max_index_size": <Maximum indexes to be processed by each worker in a Task>,
    "user_id": "root",
    "password": "msl-ssg"
  },
  
```
## Message Communication
   The data-mover application uses REQ/REP message pattern to distribute the indexing data to the
   N ClientApplications running on the N client nodes. The operation status is sent back from the 
   ClientNodes to MasterNode through PUSH/PULL message pattern. 
### Message Communication Configuration:
If the default message port doesn't work, then change it to some other ports.
```
"message": {
    "port_index": 6000,
    "port_status": 6001
  },
```   
# NFS Cluster information
Specify NFS cluster information from configuration file.
```
"nfs_config":{
               "<NFS Cluster DNS/IP address>":[ "NFS share1", "NFS share2"]
             },
```
# S3 Client Library
The supported S3 client libraries minio-python-lib, dss-client-lib or boto3.
minio-python-lib => "minio"
dss-client-lib => "dss_client"
boto3 => "boto3" 
Update the client library in the following location of configuration file.
```
"s3_storage": {
    "minio": {"url": "202.0.0.103:9000", "access_key": "minio", "secret_key": "minio123"},
    "bucket": "bucket",
    "client_lib": "dss_client" <<== Update client library, supported values [ "minio" | "dss_clint" | "boto3" ]
  },
```
# Operation
Supported operations are PUT/DEL/GET
## Operation PUT
  The PUT operation upload files into S3 storage for all NFS shares. The indexing is done in MasterNode.
  The actual upload is performed in client nodes. The index distribution is done through a IndexMonitor.
  It distribute indexes in round-robin fashion in all the clients. The actual file upload is done at the
  client node.
## Operation LIST
  The LIST operation list all keys under a provided prefix. It is performed by MasterApplication
  only. Each prefix is processed by independent workers. Results are en-queued to IndexQueue for
  DEL/GET operation. Else, gets dumped into a local file (#TODO).
## Operation DEL
  The DEL operation is dependent on LIST operation. The object keys from the LISTing operation gets 
  among the client-nodes in round-robin fashion. The actual DELETE operation is performed by ClientApplication.
  If a prefix is specified from command line, then object_keys under that prefix should be removed.
  The object prefix should be ended with forward slash (/) such as bird/ , bird/bird1/
  ```
    python3 master_application.py DEL -c 10.1.51.2 --prefix bird/
    python3 master_application.py DEL -c 10.1.51.2 
  ```
  ## Operation GET
  The GET operation is dependent on LIST operation. The object keys from the LISTing operation gets 
  among the client-nodes in round-robin fashion. The actual GET operation is performed by ClientApplication.
  The destination path should be provided from command line as sub-command along with GET. 
  ```
    python3 master_application.py GET -c 10.1.51.2 --dest_path <destiniation path>
  ```
  If prefix is not specified then accepts all the NFS shared mentioned in the configuration file as prefix.
  
 # NFS Cluster Setup
 ## Server Setup:
 ```
# NFS lib Installation 
    yum –y install nfs-utils libnfsidmap
    
# Start the services:
        systemctl enable rpcbind
        systemctl enable nfs-server
        systemctl start rpcbind
        systemctl start nfs-server
# The NFS server require to provide access of NFS shares to the client nodes.
	“/etc/exports” file has to be updated as below.
<NFS Shared Directory> <Client IP>(opt1,opt2…)

/bird 10.1.51.2(rw,sync,no_subtree_check) 10.1.51.91(rw,sync,no_subtree_check) 10.1.51.132(rw,sync,no_subtree_check) 10.1.51.107(rw,sync,no_subtree_check) 10.1.51.238(rw,sync,no_subtree_check)
/dog 10.1.51.2(rw,sync,no_subtree_check) 10.1.51.91(rw,sync,no_subtree_check) 10.1.51.132(rw,sync,no_subtree_check) 10.1.51.107(rw,sync,no_subtree_check) 10.1.51.238(rw,sync,no_subtree_check)
/cat 10.1.51.2(rw,sync,no_subtree_check) 10.1.51.91(rw,sync,no_subtree_check) 10.1.51.132(rw,sync,no_subtree_check) 10.1.51.107(rw,sync,no_subtree_check) 10.1.51.238(rw,sync,no_subtree_check)


o  Restart export “exportfs –r”
o  Show shared mount paths “exportfs -v”
Client Setup:
o  NFS lib Installation:
	yum –y install nfs-utils libnfsidmap
o  Start the service
    systemctl enable rpcbind
	systemctl start rpcbind
o  All client node should mount them before accessing shared directory.
   Create local directory to be mapped to remote directory.(optional)
   sudo mkdir /nfs-shared1”
o Mount the remote shared paths as below.
    mount <nfs node ip address>:/<nfs Shared Directoyr>  /nfs-shaed1”
o Show the remote mounts “df –kh”
```
 
  

