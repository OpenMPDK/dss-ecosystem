from logging import exception
import torch
import dss
import os
import math
import time
import argparse

from PIL import Image, ImageFile
# avoid data stream broken issue
ImageFile.LOAD_TRUNCATED_IMAGES = True
class DSSDataset(torch.utils.data.Dataset):
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
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single process data loading
            tmp_folder = "master"
            worker_id = 0
        else:
            worker_id = worker_info.id
            tmp_folder = "worker_{}".format(worker_id)
        
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)
        key = self.keys[idx]
        file_path = os.path.join(tmp_folder, key)
        try:
            client_idx = int(worker_id%self.client_num)
            client = self.clients[client_idx]
            client.getObject(key, file_path)
            if self.load2memory:
                image = Image.open(file_path)
                os.remove(file_path)
                if self.transforms:
                    np_img = self.transforms(image)
                target = int(key.split(".")[0].split("_")[-1])
                instance = np_img, target 
                return instance
            return 0, 0
        except Exception as e:
            if worker_info is None:
                print("master: error in reading {}: {}".format(key, e))
            else:
                print("worker{}: error in reading {}: {}".format(worker_id, key, e))
    
    def create_dss_clients(self, client_num):
        # return a list of dss client instances
        start = time.time()
        clients = [dss.createClient(self.endpoint, self.access_key, self.secret_key, self.dss_option) for i in range(client_num)]
        print("initializing clients: {}".format(time.time()-start))
        return clients

    def list_keys(self, prefix, limit=100000000, save2local=False):
        try:
            client = self.clients[0]
            keys = client.getObjects(prefix, limit=limit)
            if save2local:
                with open("{}-keys".format(prefix), "w") as key_file:
                    for key in keys:
                        key_file.write("{}\n".format(key))
            return keys
        except Exception as e:
            print("error in listing {}: {}".format(prefix, e))

