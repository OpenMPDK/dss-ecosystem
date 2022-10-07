import os
import sys
import time
import tensorflow as tf
import cv2
import random
import numpy as np
from tqdm import tqdm
from imutils import paths
from datetime import datetime

from s3_client import S3
# from dss_client import DssClientLib
import glob
# from utils.utility import exception

import torch
from torch.utils.data import Dataset
import torch.multiprocessing
from utils.utility import validate_s3_prefix, exec_cmd
from multiprocessing import Queue, Value
from worker import Worker

torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(3704)


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
        self.config_dataset = config["dataset"][config["dataset"]["choice"]]
        self.categories = self.config_dataset["label"]
        self.image_dimension = self.config_dataset["image_dimension"]   # height, width of image
        self.avg_image_size = self.config_dataset["avg_image_size"]     # avg disk space occupied by an image
        self.max_workers = self.config["execution"]["workers"]
        self.data_loader_workers = self.config["framework"]["PyTorch"]["DataLoader"]["num_workers"]
        self.max_object_size = int(self.config["framework"]["max_object_size"])
        self.instance_id = self.config["framework"]["instance_id"]
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
        self.dataset_size_in_bytes = Value('l', 0)  # In bytes

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
        # self.logger.info("WorkerID: {}, {}".format(worker_info.id,worker_info))
        image_name_label = self.images[index]
        image_label = image_name_label[1]
        image_ndarray, img_read_time = self.data_source(image=image_name_label,
                                                        worker_id=worker_info.id)  # Already converted using cv2.imread

        if self.transform is not None:
            image_ndarray = self.tansform(image_ndarray)

        return image_ndarray, image_label, img_read_time

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
                index += 1
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
        while self.workers_finished.value < self.max_workers:
            while self.image_queue.qsize() > 0:
                category_images = self.image_queue.get()
                listed_files = len(category_images)
                total_listed_file += listed_files
                self.images.extend(category_images)
        end_listing_time = time.monotonic()
        if not self.image_queue:
            self.logger.fatal("Couldn't list files, exit application")
            sys.exit()
        random.shuffle(self.images)
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

        start_time = time.monotonic()
        img_ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read using CV2
        time_delta = time.monotonic() - start_time

        img_ndarray = cv2.resize(img_ndarray, self.image_dimension)
        img_ndarray = torch.tensor(img_ndarray).unsqueeze(0).numpy()

        with self.dataset_size_in_bytes.get_lock():
            self.dataset_size_in_bytes.value += int(self.avg_image_size)

        return img_ndarray, time_delta

    def read_s3_object(self, **kwargs):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        For dss_client allocate memory at the application and let client library update object into that memory.

        :param object_key:
        :return:
        """
        image = kwargs["image"]
        worker_id = kwargs["worker_id"]
        object_key = image[0]
        image_buffer = None
        image_2darray = []
        try:
            if self.s3_config["client_lib"]["name"] == "dss_client":
                # image_buffer = np.asarray(bytearray( self.max_object_size )) # Used for getObjectNumpyBuffer
                image_buffer = bytearray(self.max_object_size)  # Use that for getObjectBuffer
                buffer_length = self.s3_clients[worker_id].getObject(bucket=self.s3_config["bucket"], key=object_key,
                                                                     memory=image_buffer)
                # image_numpy_array = image_buffer
                image_numpy_array = memoryview(image_buffer)[0:buffer_length]
                image_numpy_array = np.asarray(image_numpy_array)
            else:
                image_buffer, buffer_length = self.s3_clients[worker_id].getObject(bucket=self.s3_config["bucket"],
                                                                                   key=object_key)
                image_numpy_array = np.asarray(bytearray(image_buffer))

        except Exception as e:
            self.logger.excep(f"Exception:{e}")
        if buffer_length:
            with self.dataset_size_in_bytes.get_lock():
                self.dataset_size_in_bytes.value += int(buffer_length / 1024)

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
            ret, console = exec_cmd(command, True, True)
            if ret == 0:
                self.logger.info("Cleared page cache ...")
            else:
                self.logger.error("Unable to clear pagecache - ret:{},console:{}".format(ret, console))
            self.data_dirs = storage_config[self.storage_format][storage_config[self.storage_format]["choice"]]["data_dir"]
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
                client_id = str(self.instance_id) + str(i)
                s3_client = DssClientLib(credentials=self.credentials, config=self.s3_config["client_lib"].get("dss_client", {}), uuid=client_id,
                                         logger=self.logger)
                self.s3_clients.append(s3_client)
        elif self.s3_config["client_lib"]["name"] == "boto3":
            self.s3_clients = [S3(storage_name=self.storage_name, credentials=self.credentials, logger=self.logger) for
                               i in range(max_s3_client_count)]


class PythonReadDataset(RandomAccessDataset):

    def __init__(self, transforms=None, config={}, logger=None):
        super(PythonReadDataset, self).__init__(transform=transforms,
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
            if self.s3_config["client_lib"]["name"] == "dss_client":
                image_buffer = bytearray(self.max_object_size)
                buffer_length = self.s3_clients[worker_id].getObject(bucket=self.s3_config["bucket"], key=object_key,
                                                                     memory=image_buffer)
            else:
                image_buffer, buffer_length = self.s3_clients[worker_id].getObject(bucket=self.s3_config["bucket"],
                                                                                   key=object_key)
        except Exception as e:
            self.logger.excep(f"{e}")
        if buffer_length:
            with self.dataset_size_in_bytes.get_lock():
                self.dataset_size_in_bytes.value += int(buffer_length / 1024)

        return torch.FloatTensor(3, 2)


class PythonReadDatasetToDevNull(RandomAccessDataset):

    def __init__(self, transforms=None, config={}, logger=None):
        super(PythonReadDatasetToDevNull, self).__init__(transform=transforms,
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
            self.dataset_size_in_bytes.value += 1
        return torch.FloatTensor(3, 2)


class TorchImageClassificationDataset(RandomAccessDataset):
    """
    User defined dataset class. The base class has all default functionalities
    Each functions in the override section can be overide
    """

    def __init__(self, transforms=None, config={}, logger=None):
        super(TorchImageClassificationDataset, self).__init__(transform=transforms,
                                                              config=config,
                                                              logger=logger)

    # def read_file_system_data(self, image):
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

    # def read_s3_object(self, **kwargs):
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
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def __iter__(self):
        pass


class TorchObjectDetectionDataset(RandomAccessDataset):
    """
    User defined dataset class. The base class has all default functionalities
    Each functions in the override section can be overidden
    """

    def __init__(self, transforms=None, config={}, logger=None):
        super(TorchObjectDetectionDataset, self).__init__(transform=transforms,
                                                          config=config,
                                                          logger=logger)
        self.tensors = self.get_tensors()

    def __getitem__(self, index):

        image = self.tensors[0][index]
        label = self.tensors[1][index]
        bbox = self.tensors[2][index]
        img_read_time = self.tensors[3][index]

        # transpose the image such that its channel dimension becomes the leading one,
        # as accepted by the model to be used.
        image = image.permute(2, 0, 1)

        # check to see if we have any image transformations to apply and if so, apply them
        if self.transform:
            image = self.transform(image)

        with self.dataset_size_in_bytes.get_lock():
            self.dataset_size_in_bytes.value += int(self.avg_image_size)

        # return a tuple of the images, labels, and bounding box coordinates
        return image, label, bbox, img_read_time

    def __len__(self):
        return self.tensors[0].size(0)

    def get_tensors(self):
        self.logger.info("\nReading all the different objects from each of those listed images....")

        # grab the image, label, and its bounding box coordinates

        images, labels, bboxes, read_times = [], [], [], []
        for dirs in (
                self.config["storage"][self.config["storage"]["format"]][self.config["storage"][self.config["storage"]["format"]]["choice"]]["data_dir"]):
            for path in tqdm([f for f in paths.list_files(dirs, validExts='.txt')]):
                img_file = [file for file in paths.list_files(dirs, validExts='.jpg') if
                            path.split('.')[0].split('/')[-1] in file][0]

                with open(path) as file:
                    rows = file.read()
                    rows = rows.strip().split("\n")

                for row in rows:
                    row = row.split(' ')
                    (label, XMin, YMin, XMax, YMax) = row

                    label = self.categories.index(str(label))

                    start = time.monotonic()
                    image = cv2.imread(img_file)
                    read_time = (time.monotonic() - start)
                    (h, w) = image.shape[:2]

                    # scale the bounding box coordinates relative to the spatial
                    # dimensions of the input image
                    startX = float(XMin) / w
                    startY = float(YMin) / h
                    endX = float(XMax) / w
                    endY = float(YMax) / h

                    # load the image and preprocess it
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, tuple(self.config_dataset["image_dimension"]))

                    images.append(image)
                    labels.append(label)
                    bboxes.append((startX, startY, endX, endY))
                    read_times.append(read_time)

        images = np.array(images, dtype="float32")
        labels = np.array(labels)
        bboxes = np.array(bboxes, dtype="float32")
        read_times = np.array(read_times, dtype="float32")
        self.logger.info("Individual lists converted to NumPy arrays....")

        # convert NumPy arrays to PyTorch tensors
        images, labels, bboxes, read_times = torch.tensor(images), torch.tensor(labels), torch.tensor(bboxes), torch.tensor(read_times)
        self.logger.info("NumPy arrays converted to individual tensors...")

        return images, labels, bboxes, read_times


class CustomDataset(object):
    def __init__(self, transforms=None, config={}, logger=None):
        self.config = config
        self.logger = logger
        self.name = self.config["dataset"]["choice"]
        self.random_access_dataset = self.config["dataset"][self.name]["access"]["random"]
        self.class_name = self.get_class_name()
        self.dataset = None
        self.transforms = transforms
        # self.logger.info(self.class_name)

    def get_class_name(self):
        """
        Convert the string class name to actual class
        :return:
        """
        try:
            return eval(self.name)  # Convert the string to class name.
        except NameError as e:
            self.logger.execp("ERROR: Custom dataset doesn't exist! {}".format(e))
            sys.exit()

    def get_dataset(self):
        self.logger.info("INFO: Using custom dataset - {}->{}".format(self.name, self.class_name))
        if self.random_access_dataset:
            return self.class_name(self.transforms, self.config, self.logger)
        else:
            # This section should be for sequential read
            pass
