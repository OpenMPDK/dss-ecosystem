# TESS Pytorch Data Loader 

There are three files to consider: main_dss.py, dss_dataloader.py, and model.py.

The main file to run is main_dss.py which instantiates a DSS dataloader object to read the images and creates a variational autoencoder model to train using the loaded images.

In order to run the code:
    run python3 main_dss.py

There are a number of arguments that can be handled in the run:
-w: the number of workers, default is 0.
-b: the batch size, default is 16.
-c: the number of dss client instances, default is 1.
-e: the number of epochs for training, default is 1.


# DSS Dataset
In dss_dataloader.py, we design a new dataset class on top of torch.utils.data.Dataset, to abstract objects from S3 storage system (DSS).

## DataPath
DSSDataset implements a "__getitem__()" function for Dataloader instances to load data in batch. Specifically, users specify the number of dataloader workers (worker) in Dataloader initilization. Each worker calls "__getitem__()" function to load a single object and returns a batch of objects to a queue, where the dataloader master reads batch. DSSDataset initialzes a group of dss_client instances (client), then worker communicate with DSS over client. It's suggested to set the number of workers as multiplication of the number of clients.

## Initialization
DSSDataset needs arguments of dss backend info (i.e., access key, secret key, and dss option), the number of clients and the transoform function, to initialize a DSSDataset instance. If load2memory is False, worker only downloads data from S3 to local file system without loading data into memory.
```
    def __init__(self, prefix, access_key, secret_key, endpoint, dss_option, client_num, transforms=None, load2memory=True):
        super(DSSDataset).__init__()
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.dss_option = dss_option
        self.client_num = client_num
        # client_num should be a divisor of the number of workers
        self.clients = self.create_dss_clients(client_num)
        assert(len(self.clients)>0)
        self.keys = list(self.list_keys(prefix=prefix, save2local=False))
        self.transforms = transforms
        self.load2memory = load2memory
```

# Example of minio parameters
'''
access_key = "myminio"
secret_key = "test"
endpoint = "http://100.1.1.100:9000"
option = dss.clientOption()
option.maxConnections = 1
'''

# VAE model

In model.py, we have a variational autoencoder model which is an AI usecase that we use in order to test the performance of our DSS stack.