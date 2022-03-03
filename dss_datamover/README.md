# DSS Data Mover

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

In these parameters,  ```datamover_logging_path , datamover_conf_dir ``` are used to set the path to the datamover config and log files. 

It is also required to keep ```datamover_client_lib: dss_client``` as is. 

```datamover_message_port_index , datamover_message_port_status``` can be modified based on the machine's available port, the default port is 4000. 

Parameters such as ```datamover_master_size``` is not needed to be set.

Table below shows the tunable parameters.

| Parameter                       | Description   |
| --------------------------------|-------------- |
| datamover_master_workers        | Number of workers does parallel indexing and listing. Based on the core available on data mover master, tune the number of master workers to improve the performance |
| datamover_master_max_index_size | Per each master worker how many file indexes can be handled simultaneously, on a physical server this value can be >= 1000, meaning each client worker can process maximum 1000 files sequentially to S3 storage|
| datamover_client_workers        | Number of client workers performing parallel I/O operation on a single node, the value can be tuned based on available cores on each client|
| datamover_client_max_index_size | Maximum number of file index can be passed through an index message to the different client-application, on a physical system this value can be >= 1000 |
| datamover_nfs_shares:           | The ip address of the NFS server and their share directories are set, respectively.|
| datamover_logging_level: INFO   | The log level INFO, DEBUG, WARNING of data mover is specified.|

### Deployment

After Deploying the DSS software using the ansible playbook, Datamover is already deployed

```
ansible-playbook -i inv_file  playbooks/deploy_dss_software.yml 
```

In case you change datamover configuration after deployment of the dss_software, datamover should be deployed again.

```
ansible-playbook -i inv_file  playbooks/deploy_datamover.yml 
```

### Start DataMover

Start DataMover using start_start_datamover.yml playbook to move the data from NFS server and PUT it to DSS storage servers.

```
ansible-playbook -i inv_file  playbooks/start_datamover.yml 
```

to test that the data is moved from NFS to DSS storage on client/storage server
data is generally stored under dss<em>i</em> bucket which *i* specifies the cluster's index and *dss* bucket will keep the datamover configuration file.

```
cd /usr/dss/nkv-datamover
./mc ls autominio 
./mc ls autominio/dss0
```

### I/O Operations

Using the start_datamover playbook we can execute I/O operations (GET, DEL, LIST):

```
ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=GET"

ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=DEL"

ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=LIST"
```

To test DM uploaded the files into S3 storage can be checked as below.
- Data mover LIST operation to indicate how many objects have been uploaded
- Checking Dry Run datamover I/O operation without uploading files.
- Checking Data Integrity that the file uploaded on S3 is the same as data on NFS. 

### Dry Run

Dry Run option is to exercises the data mover I/O operation without actually calling the s3 functions. Read files from NFS shares, but skip the upload operation. Show RAW NFS read performance. Dry run in PUT operation access the data on the NFS server copy it to the buffer and then without writing the data to DSS Storage will skip the s3 call and delete the buffer content. Dry run GET/DEL is dependent on parallel listing operation. It performs listing and exercise all the part of code except calling S3 download / remove object.

```
ansible-playbook -i your_inventory playbooks/start_datamover.yml -e "datamover_dryrun=true"
```

### Data Integrity Test

The Data Integrity test is to make sure the data uploaded on DSS is the same as the data on the NFS. When the option is enabled, data mover performs a checksum validation test of all objects on object store. It also performs data-integrity check for uploaded data when “—skip_upload” option is specified.

```
ansible-playbook -i inv_file playbooks/start_datamover.yml -e "datamover_operation=TEST"
```

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
mount <nfs node ip address>:/<nfs Shared Directory>  /nfs-shared1”           
```

verify mounted shared directory

```df –kh```

### I/O Operations
- Running PUT operation with compaction option, optimizes the performance for GET and LIST of objects on DSS.
- LIST/GET/DEL operations can be performed either recursively for the whole data on DSS or only for a specific prefix. 
- In case of PUT failure, with warning ```WARNING: DataMover RESUME operation is required!```, please retry the put operation again.  
- The default path for Data mover config file ```/usr/dss/nkv-datamover/config/config.json``` to specify the config path use --config option. 
 
 ```--config {{ datamover_conf_dir }}/config.json```

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
  sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py TEST --data_integrity –skip_upload --dest_path <destiniation path>'
  sudo sh -c ' source  /usr/local/bin/setenv-for-gcc510.sh && python3 master_application.py PUT --dryrun'

```

### NFS Share Guideline
- For a better perfromance, it is advised to put the number of files per directory, same or slightly more than master max index. For example, if there are 10,000 files under a directory , and that directory get processed by a worker W1, you should be getting 10 message if mas_index_size is set to 1000. 
You may chose to set 2000, which 'll send out 5 message and that may speed up process further. 
On another note make sure that in nested directory each directory has reasonable number of files (same or slightly more than master max index).

- Here is an example mounted multiple NFS share on client server and one of its NFS server
```
[ansible@client]$ df -h
Filesystem                                   Size  Used Avail Use% Mounted on
nfs.srv1.ip:/mnt/nfs_share/5gb           985G  224G  711G  24% /srv1/mnt/nfs_share/5gb
nfs.srv2.ip:/mnt/nfs_share/10gb-B        246G  207G   27G  89% /srv2/mnt/nfs_share/10gb-B
nfs.srv2.ip:/mnt/nfs_share/5gb-B         246G  207G   27G  89% /srv2/mnt/nfs_share/5gb-B
nfs.srv1.ip:/mnt/nfs_share/15gb          985G  224G  711G  24% /srv1/mnt/nfs_share/15gb
nfs.srv1.ip:/mnt/nfs_share/10gb          985G  224G  711G  24% /srv1/mnt/nfs_share/10gb
nfs.srv2.ip:/mnt/nfs_share/15gb-B        246G  207G   27G  89% /srv2/mnt/nfs_share/15gb-B
```

```
[ansible@nfs.srv1.ip ~]$ ls /mnt/nfs_share/
10gb  15gb   5gb  
[ansible@nfs.srv1.ip ~]$ls /mnt/nfs_share/5gb
0_0  1MB_0001.dat  1MB_0003.dat  1MB_0005.dat  1MB_0007.dat  1MB_0009.dat  1MB_0011.dat  1MB_0013.dat  1MB_0015.dat  1MB_0017.dat  1MB_0019.dat
0_1  1MB_0002.dat  1MB_0004.dat  1MB_0006.dat  1MB_0008.dat  1MB_0010.dat  1MB_0012.dat  1MB_0014.dat  1MB_0016.dat  1MB_0018.dat  1MB_0020.dat
```

