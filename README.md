# TESS Data Mover

## (I) Datamover Deployment Using Ansible

### Configuration

The default configuration for the data mover is being set in *deploy/roles/create_datamover_conf/defaults/main.yml*. To customize the datamover configuration, user should uncomment and set the parameters at *deploy/group_var/all.yml* file under __### Datamover Settings__ block.
Below as example of shown.

```
### Datamover Settings                                                     
# datamover_conf_dir: /etc/dss/datamover                   
# datamover_master_workers: 5                             
# datamover_master_max_index_size: 500                      
# datamover_master_size: 1GB
# datamover_client_workers: 5
# datamover_client_max_index_size: 500
# datamover_client_user_id: ansible
# datamover_client_password: ansible
# datamover_message_port_index: 4000
# datamover_message_port_status: 4001
# datamover_nfs_shares:
#   - ip: 192.168.200.199
#     shares:
#       - /mnt/nfs_share/5gb
#       - /mnt/nfs_share/10gb
#       - /mnt/nfs_share/15gb
#   - ip: 192.168.200.200
#     shares:
#       - /mnt/nfs_share/5gb-B
#       - /mnt/nfs_share/10gb-B
#       - /mnt/nfs_share/15gb-B
# datamover_master_size: bucket
# datamover_client_lib: dss_client
# datamover_logging_path: /var/log/dss
# datamover_logging_level: INFO
```

In these parameters,  ```datamover_logging_path , datamover_conf_dir ``` used to set the path to the datamover config and logging. It is also required to keep ```datamover_client_lib: dss_client``` as is. ```datamover_message_port_index , datamover_message_port_status``` can be modified based on the machine's available port, the default port is 4000. Parameters such as ```datamover_master_size, datamover_master_size``` is not needed to be set.
Table below shows the tunable parameters.

| Parameter                       | Description   |
| --------------------------------|-------------- |
| datamover_master_workers        | Based on the core available on data mover master, tune the number of master workers to improve the performance |
| datamover_master_max_index_size | Per each master worker how many indexes can be handled simultaneously, on a physical server this value can be >= 1000|
| datamover_client_workers        | The number of worker each client uses to perform the I/O operation, the value can be tuned based on available cores on each client|
| datamover_client_max_index_size | Per each client worker how many indexes can be handled simultaneously for I/O operations, on a physical system this value can be >= 1000 |
| datamover_nfs_shares:           | The ip address of the NFS server and their share directories are set, respectively.|
| datamover_logging_level: INFO   | The log level INFO, DEBUG, WARNING of data mover is specified.|

### Deployment

After Deploying the DSS software using the ansible playbook, Datamover is already deployed

```ansible-playbook -i inv_file  playbooks/deploy_dss_software.yml ```

In case you change datamover configuration after deployment of the dss_software, datamover should be deployed again.

```ansible-playbook -i inv_file  playbooks/deploy_datamover.yml ```

### Start DataMover

Start DataMover using start_start_datamover.yml playbook to move the data from NFS server and PUT it to TESS storage servers.

```ansible-playbook -i inv_file  playbooks/start_datamover.yml ```

to test that the data is moved from nfs to TESS storage on client/storage server
data is generally stored under dss<em>i</em> bucket which *i* specifies the cluster's index and *dss* bucket will keep the datamover configuration file.

```
cd /usr/dss/nkv-datamover
./mc ls autominio 
./mc ls autominio/dss0
```

### I/O Operations

Using the start_datamover playbook we can execute I/O operations (GET, DEL, LIST):
```ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=GET"```
```ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=DEL"```
```ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=LIST"```

### Dry Run

Dry Run option is to exercises the data mover I/O operation without actually calling the s3 functions. Read files from NFS shares, but skip the upload operation. Show RAW NFS read performance. Dry run in PUT operation access the data on the NFS server copy it to the buffer and then without writing the data to TESS Storage will skip the s3 call and delete the buffer content. Dry run GET and DEL is initiate indexing on the master and distribute the work across the clients, but no GET and DEL is happening.

```ansible-playbook -i your_inventory playbooks/start_datamover.yml -e "datamover_dryrun=true"```

### Data Integrity Test

The Data Integrity test is to make sure the data uploaded on TESS is the same as the data on the NFS. When the option is enabled, data mover Performs a checksum validation test of all objects on object store. It starts indexing and then uploading data for each prefix through a worker from a client
node. During that process, it keeps track of file and corresponding md5sum hash in a buffer. Subsequently, a GET
operation is initiated with the same prefix which downloads files in the temporary destination path and compares md5sum
with corresponding file key in buffer.

```ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=TEST"```

## (II) Data Mover Mannual Deployment

### NFS Cluster Setup

#### NFS Server Setup

Install NFS Lib on NFS server and enable then start the rpcbind and nfs-server services.

```
yum –y install nfs-utils libnfsidmap
systemctl enable rpcbind
systemctl enable nfs-server
systemctl start rpcbind
systemctl start nfs-server

```

The NFS server provides access of NFS shares to the client nodes via “/etc/exports” file with the following format:
``` <NFS Shared Directory> <Client IP>(opt1,opt2…) ```

for example:

```
/bird 10.1.51.2(rw,sync,no_subtree_check) 10.1.51.91(rw,sync,no_subtree_check) 10.1.51.132(rw,sync,no_subtree_check) 10.1.51.107(rw,sync,no_subtree_check) 10.1.51.238(rw,sync,no_subtree_check)
/dog 10.1.51.2(rw,sync,no_subtree_check) 10.1.51.91(rw,sync,no_subtree_check) 10.1.51.132(rw,sync,no_subtree_check) 10.1.51.107(rw,sync,no_subtree_check) 10.1.51.238(rw,sync,no_subtree_check)
```

Restart the export file
``` exportfs –r ```

verify shared mount paths

``` exportfs -v ```

#### NFS Client Setup

Install the NFS lib
``` yum –y install nfs-utils libnfsidmap ```

enable and start the rpcbind service

```
systemctl enable rpcbind
systemctl start rpcbind
```
all client should mount NFS shared directory before accessing them.

```
sudo mkdir /nfs-shared1     # Create local directory to be mapped to remote directory.(optional)           
mount <nfs node ip address>:/<nfs Shared Directoyr>  /nfs-shaed1”           
```

verify mounted shared directory

```df –kh```

### I/O Operations

```
cd /usr/dss/nkv-datamover
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT  '
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT  --compaction'
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py LIST --dest_path <Destiniation Path> '
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py LIST  --prefix <prefix>/ --dest_path <Destiniation Path>'
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL '
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py DEL --prefix bird/'
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --dest_path <Destination Path>'
sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py GET --prefix bird/ --dest_path <Destination Path>'
```

### Data Integrity Test and Dry Run

Here are the examples to run data mover with integrity test and dry run options:

```
  sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py TEST --data_integrity --dest_path <destiniation path>'
  sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py TEST --unit --dest_path <destiniation path>'
  sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT --dryrun'

```