import os,sys
import tensorflow as tf
import cv2
import random
import numpy as np
from datetime import datetime
from s3_client import S3
#from dss_client import DssClientLib
import glob
#from utils.utility import exception

import torch
from torch.utils.data import Dataset
from utils.utility import validate_s3_prefix, exec_cmd
from multiprocessing import Queue, Value
from worker import Worker
import time


class RandomAccessDataset(Dataset):
    """
    MapStyle datset has following three methods __init__, __len__, __getitem__.

    return size of the dataset and indexing
    """

    def __init__(self, transform=None, config={}, logger=None):
        self.transform = transform
        self.config = config
        self.logger = logger

        # Read config
        self.config_dataset = config["dataset"]
        self.categories = self.config_dataset["label"]
        self.image_dimension = self.config_dataset["image_dimension"]  # height, width of image
        self.max_workers = self.config["execution"]["workers"]
        self.data_loader_workers = self.config["framework"]["PyTorch"]["DataLoader"]["num_workers"]
        self.data_source = None  # Function to read data from
        self.credentials = None  # Required to access data from storage.
        self.storage_name = None  # Storage name such as aws,dss
        self.storage_format = None  # Data storage format such as s3, file system.
        self.data_dirs = None
        self.s3_clients = []
        self.s3_config = {}
        # Call initial data-source setup functions

        self.set_data_source()


        # Parallel Listing
        self.workers_finished = Value('i', 0)
        self.workers = []
        self.image_queue = Queue()
        # Listing time
        self.listing_time = 0

        # Collect all the image names and corresponding label
        self.images = []
        self.get_image_names()  # [(image1,0),(image200,1)]

        # Transform

        # Datasize calculation
        self.dataset_size_in_bytes = Value('l', 0) # In bytes

    def __len__(self):
        """
        Returns the size of dataset
        :return:
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Index dataset and return ith sample dataset[i]
        This behave like python generator.
        :return:
        """
        worker_info = torch.utils.data.get_worker_info()
        #self.logger.info("WorkerID: {}, {}".format(worker_info.id,worker_info))
        image_name_label = self.images[index]
        image_label = image_name_label[1]
        image_ndarray = self.data_source(image=image_name_label,
                                         worker_id=worker_info.id)  # Already converted using cv2.imread

        if self.transform is not None:
            image_ndarray = self.tansform(image_ndarray)

        return image_ndarray, image_label

    def get_image_names(self):
        """
        Create a image dataset

        Parallel Listing:
         - Check no of file system shares or prefixes and categories . Based on that numbers, create workers
          Shares/Prefixes = N, Categories = M , Max Workers Wmax = 50, W =?
          if NxM < Wmax:
             W = NxM
          else:
             W = Wmax
        :return: None
        """

        fs_share_count = len(self.data_dirs)  # N
        categoris_count = len(self.categories)  # M


        # Calculate maximum number of workers.
        if self.max_workers > fs_share_count * categoris_count:
            self.max_workers = fs_share_count * categoris_count

        # Create max_workers number of sets of category paths.
        category_paths = [[] for i in range(self.max_workers)]

        index = 0
        for data_dir in self.data_dirs:
            for category in self.categories:
                if index >= self.max_workers:
                    index = 0  # Reset counter
                if not data_dir.endswith("/"):
                    category_paths[index].append(data_dir + "/" + category)
                else:
                    category_paths[index].append(data_dir + category)
                index +=1
        start_listing_time = time.monotonic()
        s3_client = None
        # Distribute load among the max_workers.
        for worker_id in range(self.max_workers):
            if self.storage_format == "s3":
                s3_client = self.s3_clients[worker_id]
            w = Worker(id=worker_id,
                       s3_config=self.s3_config,
                       s3_client=s3_client,
                       storage_format=self.storage_format,
                       categories=self.categories,
                       data_dirs=category_paths[worker_id],
                       queue=self.image_queue,
                       worker_finished=self.workers_finished,
                       logger=self.logger
                       )
            w.start()
            self.workers.append(w)
        # Aggregate all files listed by workers
        self.logger.info("Started listing with {} workers".format(self.max_workers))
        total_listed_file = 0
        while self.workers_finished.value < self.max_workers :
            while self.image_queue.qsize() > 0:
                category_images = self.image_queue.get()
                listed_files = len(category_images)
                total_listed_file += listed_files
                self.images.extend(category_images)
        end_listing_time = time.monotonic()
        if not self.image_queue:
            self.logger.fatal("Couldn't list files, exit application")
            sys.exit()
        self.listing_time = "{:0.4f}".format(end_listing_time - start_listing_time)
        self.logger.info("Total files listed: {}, Time: {} seconds".format(total_listed_file, self.listing_time))

    def read_file_system_data(self, **kwargs):
        """
        Read data from file system.
        :param image:
        :return:
        """
        image = kwargs["image"]
        image_path = image[0]  # (image1,0) => (<image_name>,<Category Index>)
        img_ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read using CV2
        img_ndarray = cv2.resize(img_ndarray, self.image_dimension)
        return img_ndarray

    def read_s3_object(self, **kwargs):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        :param object_key:
        :return:
        """
        image = kwargs["image"]
        worker_id = kwargs["worker_id"]
        object_key = image[0]
        image_buffer = None
        image_2darray = []
        try:
            image_buffer = self.s3_clients[worker_id].getObject(bucket=self.s3_config["bucket"], key=object_key)
        except Exception as e:
            self.logger.excep(f"Exception:{e}")
        if image_buffer:
            with self.dataset_size_in_bytes.get_lock():
                self.dataset_size_in_bytes.value += int(len(image_buffer) / 1024)
            image_numpy_array = np.asarray(bytearray(image_buffer))
            # Converts to image format
            image_2darray = cv2.imdecode(image_numpy_array, cv2.IMREAD_GRAYSCALE)
            image_2darray = cv2.resize(image_2darray, self.image_dimension)
        return image_2darray


    def set_data_source(self):
        """
        Set the data source from configuration.
        :return: A function pointer
        """
        storage_config = self.config["storage"]
        self.storage_format = storage_config["format"].lower()
        self.storage_name = storage_config["name"].lower()
        data_source_summary = ""

        if self.storage_format == "s3":
            bucket = storage_config[self.storage_format]["bucket"]
            client_lib = storage_config[self.storage_format]["client_lib"]
            self.credentials = storage_config[self.storage_format][self.storage_name]["credentials"]
            self.s3_config = {"credentials": self.credentials, "bucket": bucket, "client_lib": client_lib}
            self.data_dirs = storage_config[self.storage_format]["prefix"]
            if client_lib["name"] == "boto3":
                data_source_summary = "Bucket:{} ".format(bucket)
            self.data_source = self.read_s3_object
            self.get_s3_clients()
            data_source_summary = ", Client_Lib:{} ".format(client_lib["name"]) + data_source_summary
        elif self.storage_format == "fs":
            # Empty page cache.
            command = "echo 3> /proc/sys/vm/drop_caches"
            ret,console = exec_cmd(command, True, True)
            if ret ==0 :
                self.logger.info("Cleared page cache ...")
            else:
                self.logger.error("Unable to clear pagecache - ret:{},console:{}".format(ret, console))
            self.data_dirs = storage_config[self.storage_format]["data_dir"]
            self.data_source = self.read_file_system_data

        self.logger.info("Data source:{}, format:{}{}".format(self.storage_name, self.storage_format, data_source_summary))

    def get_s3_clients(self):
        """
        Get s3_clients equal numbers of max_workers.
        :return: None
        """
        max_s3_client_count = self.max_workers
        if self.data_loader_workers > self.max_workers:
            max_s3_client_count = self.data_loader_workers
        self.logger.info(f"** Creating {max_s3_client_count} s3 clients for parallel processing! **")
        if self.s3_config["client_lib"]["name"] == "dss_client":
            from dss_client import DssClientLib
            for i in range(max_s3_client_count):
                s3_client = DssClientLib(credentials=self.credentials,config=self.s3_config["client_lib"]["dss_client"],
                                         logger=self.logger)
                self.s3_clients.append(s3_client)
        elif self.s3_config["client_lib"]["name"] == "boto3":
            self.s3_clients = [ S3(storage_name=self.storage_name, credentials=self.credentials, logger=self.logger) for
                                i in range(max_s3_client_count)]


class PythonReadDataset(RandomAccessDataset):

    def __init__(self, config={},logger=None):
        super(PythonReadDataset, self).__init__(
                                                transform=None,
                                                config=config,
                                                logger=logger)

    def read_file_system_data(self, **kwargs):
        image = kwargs["image"]
        image_path = image[0]  # (image1,0) => (<image_name>,<Category Index>)
        # category = self.label[image[1]]  # Find out category
        # image_path = self.data_dir + "/" + category + "/" + image_name
        # img_ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read using CV2
        # print(image)
        with open(image_path, mode='rb') as file:
            fileContent = file.read()  # Returns byte object
            with self.dataset_size_in_bytes.get_lock():
                self.dataset_size_in_bytes.value += int(len(fileContent) / 1024)

        return torch.FloatTensor(3, 2)

    def read_s3_object(self, **kwargs):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        :param object_key:
        :return:
        """
        image = kwargs["image"]
        worker_id = kwargs["worker_id"]
        object_key = image[0]
        image_buffer = None
        image_2darray = []
        try:
            image_buffer = self.s3_clients[worker_id].getObject(bucket=self.s3_config["bucket"], key=object_key)
        except Exception as e:
            self.logger.excep(f"{e}")
        if image_buffer:
            with self.dataset_size_in_bytes.get_lock():
                self.dataset_size_in_bytes.value += int(len(image_buffer) / 1024)
                #self.logger.info("File Size:{} Bytes".format(self.dataset_size_in_bytes.value))

        return torch.FloatTensor(3, 2)

class PythonReadDatasetToDevNull(RandomAccessDataset):

    def __init__(self, config={}, logger=None):
        super(PythonReadDatasetToDevNull, self).__init__(
                transform=None,
                config=config,
                logger=logger)
    def read_s3_object(self, **kwargs):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        :param object_key:
        :return:
        """

        image = kwargs["image"]
        worker_id = kwargs["worker_id"]
        object_key = image[0]
        self.s3_clients[worker_id].getObjectToFile(bucket=self.s3_config["bucket"], key=object_key, dest_file_path="/dev/null")
        with self.dataset_size_in_bytes.get_lock():
            self.dataset_size_in_bytes.value += 1024
        return torch.FloatTensor(3, 2)



class TorchImageClassificationDataset(RandomAccessDataset):
    """
    User defined dataset class. The base class has all default functionalities
    Each functions in the override section can be overide
    """

    def __init__(self, config={}, logger=None):
        super(TorchImageClassificationDataset, self).__init__(
                                                              transform=None,
                                                              config=config,
                                                              logger=logger)



    #def read_file_system_data(self, image):
        """
        Example
        :param image:
        :return:
        """
        """
        image_name = image[0]  # (image1,0) => (<image_name>,<Category Index>)
        category = self.label[image[1]]  # Find out category
        #  Start time
        image_path = self.data_dir + "/" + category + "/" + image_name
        with open(image_path, "rb") as fh:
            image_buffer = fh.read()
        #image_numpy_array = np.asarray(bytearray(image_buffer))
        img_ndarray = np.asarray(bytearray(image_buffer))
        # Converts to image format
        #img_ndarray = cv2.imdecode(image_numpy_array, cv2.IMREAD_GRAYSCALE)
        # end time
        return img_ndarray
        """


    #def read_s3_object(self, **kwargs):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        :param object_key:
        :return:
        """
        """
        image = kwargs["image"]
        worker_id = kwargs["worker_id"]
        object_key = image[0]
        #print("ObjectKey:{}".format(object_key))
        self.s3_clients[worker_id].getObjectToFile(bucket=self.s3_config["bucket"], key=object_key, dest_file_path="/var/log/dss")
        image_path= "/var/log/dss/" + object_key
        with open(image_path, "rb") as fh:
            image_buffer = fh.read()
        image_numpy_array = np.asarray(bytearray(image_buffer))
        # Converts to image format
        image_2darray = cv2.imdecode(image_numpy_array, cv2.IMREAD_GRAYSCALE)
        image_2darray = cv2.resize(image_2darray, self.image_dimension)
        return image_2darray
        """

class SequentialAccessDataset(Dataset):
    """
    This is Python iterator based
    """
    def __init__(self,config, logger):
        self.config = config
        self.logger = logger

    def __iter__(self):
        pass



class CustomDataset(object):
    def __init__(self,config, logger):
        self.config = config
        self.logger = logger
        self.random_access_dataset = config["dataset"]["access"]["random"]
        self.name = config["dataset"]["name"]
        self.class_name = self.get_class_name()
        self.dataset = None
        #self.logger.info(self.class_name)


    def get_class_name(self):
        """
        Convert the string class name to actual class
        :return:
        """
        try:
            return eval(self.name) # Convert the string to class name.
        except NameError as e:
            self.logger.execp("ERROR: Custom dataset doesn't exist! {}".format(e))
            sys.exit()

    def get_dataset(self):
        self.logger.info("INFO: Using custom dataset - {}->{}".format(self.name,self.class_name))
        if self.random_access_dataset :
            return self.class_name(self.config,self.logger)
        else:
            # This section should be for sequential read
            pass
