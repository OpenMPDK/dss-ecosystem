# dss-datamover

A command line tool to interact with DSS object storage. It supports the following operations.

**PUT:** Uploads posix based file system data to DSS object storage.

**LIST:** List object keys from the object storage.

**GET:** Download objects and store in the shared / local file system.

**DEL:** Remove objects from DSS object storage.

**DataIntegrity:** Perform data-integrity with md5sum enabled.

The tool works in a distributed and standalone mode ( single node execution ).
In the distributed version, it uses a master node to initiate overall process and
a set of client nodes to perform in parallel. The remote ClientApp are spawns using
ssh connection from **paramiko** library. The "ClientApplication" running on the client node interact with "MasterApplication" running on master node via TCP socket communication.

## Dependency

Install following packages on the client nodes before launching client application to those nodes.

```bash
python3 -m pip install paramiko
python3 -m pip install boto3
python3 -m pip install pyminio
python3 -m pip install ntplib

Use isntall.sh on each master/client node.
OR
pip3 install requirements.txt
```

## Execution command

As a non-root user, please run the below commands with '''sudo''' and make sure the non-root user
is part of sudoers file.

```bash
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT  '
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT  --compaction yes'
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py LIST --dest_path <Destiniation Path> '
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py LIST  --prefix <prefix>/ --dest_path <Destiniation Path>'
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL '
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL --prefix bird/'
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --dest_path <Destination Path>'
sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --prefix bird/ --dest_path <Destination Path>'

Dry Run:
- Read files from NFS shares, but skip the upload operation. Show RAW NFS read performance
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT --dryrun' 
- It performa every steps involved in DELETE operation except actual DELETE from S3 storage.
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL --dryrun'
- It performs every steps involved in GET operation except actual S3 GET operation
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --dest_path <"Destination File Path"> --dryrun' 
NFS Cluster: 10.1.51.2

Debug:
- Use "--debug/-d" switch to run DataMover in debug mode.

Testing:
  DataIntegrity:
  - This functionality and testing checks data integrity. First DM upload data for each prefix, subsequently
   run GET and perform md5sum. During the upload process it collect md5sum hash key from each file and store them in 
   temporary buffer and during GET call it uses that to check data integrity.
   python3 master_application.py TEST --data_integrity --dest_path <Destination Path> --debug 
```

## TESS Master Node and Client Nodes

### Master Node

   The master node is the node where, the DataMover starts.
   It does indexing for PUT operation and distribute the file index to the ClientApps
   running on client nodes evenly. It has MasterApp running multiple workers to perform indexing.
   The MasterApp also consist three monitors such as IndexDistribution, StatusPoller,
   OperationProgress monitors.
   **IndexDistributionMonitor**: Distribute the file-indices or object_keys to the client nodes
   for PUT or LIST operation respectively.
   **StatusPollerMonitor**: Read status messages from each clients and add them to a status queue.
   **OperationProgressMonitor**: Read all status data from status queue and calculate the progress.
                   Display the result.
   The file upload is performed by the ClientApplication running in each TESS client node.

### MasterApplication configuration

Update master section in the config.json

```json
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

## Client Nodes

   The ClientApplication (ClientApp) running in each client node performs the actual operation PUT/DEL/GET.
   Further, the received messages are enqueued to a TaskQ. Multiple workers
   running on that client node, process each task independently and
   add send the operation status to MasterApp for aggregation of status.

### How to configure ClientApplication

Specify TESS client nodes, ip address or DNS name

```json
"tess_clients_ip":["10.1.51.91", "10.1.51.132", "10.1.51.107"],
```

Update client section in config.json

```json
 "client": {
    "workers": <Specifiy Workers>,
    "max_index_size": <Maximum indexes to be processed by each worker in a Task>,
    "user_id": "user-id",
    "password": "msl-ssg"
  },
 **Passwordless lauch of ClientApp:** User doesn't require to specify password to launch ClientApp
  on differ nodes as long as ssh-key is coppied to all the nodes where ClientApp will be launched.
  Keep the "password" section blank.  
```

## Message Communication

   The DataMover uses TCP socket for the communication between MasterApp and ClientApps.
   The communication channel is established between the apps after they were launched.
   The indices are distributed through IndexDistributionMonitor and operation status
   are aggregated through StatusPollerMonitor.

### Message Communication Configuration

   The "port_index" is used by the IndexDistributionMonitor of MasterApp and message_server_index of ClientApp.
   The "port_status" is used by the StatusPoller of MasterApp and message_server_status of ClientApp.

```json
"message": {
    "port_index": 6000,
    "port_status": 6001
  },
```

**Common Issues:** If the default message port doesn't work, then change it to some other ports.

## NFS Cluster information

Specify NFS cluster information from configuration file.

```json
"nfs_config":{
               "<NFS Cluster DNS/IP address>":[ "NFS share1", "NFS share2"]
             },
```

The following CLI args are also supported
```
--nfs-server
--nfs-port
--nfs-share
```

Command using NFS CLI args
```
sh -c "source  /usr/local/bin/setenv-for-gcc510.sh  && python3 /usr/dss/nkv-datamover/master_application.py  PUT --config /etc/dss/datamover/config.json  --compaction yes --nfs-server <nfs server ip> --nfs-port <nfs port #> --nfs-share <path to nfs share>"
```

## S3 Client Library

The supported S3 client libraries minio-python-lib, dss-client-lib or boto3.
minio-python-lib => "minio"
dss-client-lib => "dss_client"
boto3 => "boto3"
Update the client library in the following location of configuration file.

```json
"s3_storage": {
    "minio": {"url": "202.0.0.103:9000", "access_key": "minio", "secret_key": "minio123"},
    "bucket": "bucket",
    "client_lib": "dss_client" <<== Update client library, supported values [ "minio" | "dss_clint" | "boto3" ]
  },
```

## Operation

Supported operations are PUT/DEL/GET

## Operation PUT

  The PUT operation upload files into S3 storage for all NFS shares. The indexing is done in MasterNode.
  The actual upload is performed in client nodes. The index distribution is done through a IndexMonitor.
  It distribute indexes in round-robin fashion in all the clients. The actual file upload is done at the
  client node.

### Target Compaction

  The DSS target compaction is by default initiated. The compaction can be toggled off by overriding using `"--compaction no"` CLI arg
  along with regular upload command, or by specifying the compaction field in the config file with the desired compaction option `"compaction": "no"`

  ```json
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT --compaction yes'
  ```

  Config file
  ```
  "compaction": "yes"
  ```

### Partial upload of data from a NFS share

  The partial data can be uploaded to S3 with proper prefix. A valid prefix should have following signature.
  A prefix should start with "NFS" server "ip_address" and end with a forward slash "/".
  `<nfs_server_ip>/<prefix>/`

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT 
              --prefix <nfs_server_ip>/<prefix path>/ '
  ```

#### Configuration

  Include IP address of the DSS targets from the DataMover configuration file.

  ```json
  "dss_targets": ["202.0.0.104"]  # Here "202.0.0.104" is the ip address of the DSS target.
  ```

## Operation LIST

  The LIST operation list all keys under a provided prefix. It is performed by MasterApplication
  only. Each prefix is processed by independent workers. Results are en-queued to IndexQueue for
  DEL/GET operation. Else, gets dumped into a local file.

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py LIST --dest_path <Destiniation Path> '
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py LIST  --prefix <prefix>/ --dest_path <Destiniation Path>'
  ```

## Operation DEL

  The DEL operation is dependent on LIST operation. The object keys from the LISTing operation gets
  among the client-nodes in round-robin fashion. The actual DELETE operation is performed by ClientApplication.
  If a prefix is specified from command line, then object_keys under that prefix should be removed.
  The object prefix should be ended with forward slash (/) such as bird/ , bird/bird1/

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL --prefix bird/'
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL ' 
  ```

## Operation GET

  The GET operation is dependent on LIST operation. The object keys from the LISTing operation gets
  among the client-nodes in round-robin fashion. The actual GET operation is performed by ClientApplication.
  The destination path should be provided from command line as sub-command along with GET.

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --dest_path <destiniation path>'
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --dest_path <destiniation path> --prefix bird/'
  ```

  If prefix is not specified then accepts all the NFS shared mentioned in the configuration file as prefix.

## Operation TEST

  The DataMover test can be initiated with TEST operation. The data integrity test can be initiated as below.

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py TEST --data_integrity --dest_path <destiniation path>'
  ```

  The DataIntegrity test, first start indexing and start uploading data for each prefix through a worker from a client
  node. During that process, it keep track of file and corresponding md5sum hash in a buffer. Subsequently, a GET
  operation is initiated with same prefix and object keys which downloads files in the temporary destination path and compares md5sum
  with corresponding file key in buffer.
  
  A prefix based data_integrity test should be executed as below. User should provide a prefix
  that start without forward slash and ending with forward slash `**<nfs_server_ip>/<prefix path>/**`.

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py TEST --data_integrity --dest_path <destiniation path> --prefix <prefix/>'
  ```

  Data-integrity can be performed on pre-existing S3 data uploaded through DataMover. If \"--skip_upload" switch is
  specified then, DM skip uploading files under the specified prefix.

  ```bash
  sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py TEST --data_integrity --dest_path <destiniation path> --skip_upload'
  ```

## NFS Cluster Setup

## Server Setup

 ```bash
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
