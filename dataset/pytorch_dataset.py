import os
import tensorflow as tf
import cv2
import random
import numpy as np
from datetime import datetime
from s3_client import S3
#from utils.utility import exception

import torch
from torch.utils.data import Dataset
from utils.utility import validate_s3_prefix



class RandomAccessDataset(Dataset):
    """
    MapStyle datset has following three methods __init__, __len__, __getitem__.

    return size of the dataset and indexing
    """

    def __init__(self, transform=None, config={}):
        self.transform = transform
        self.config = config

        # Read config
        self.config_dataset = config["dataset"]
        self.label = self.config_dataset["label"]
        self.image_dimension = self.config_dataset["image_dimension"]  # height, width of image
        self.data_source = None  # Function to read data from
        self.credentials = None  # Required to access data from storage.
        self.storage_name = None  # Storage name such as aws,dss
        self.storage_format = None  # Data storage format such as s3, file system.
        self.data_dir = None
        self.s3_client = None
        self.bucket = None
        # Call initial data-source setup functions

        self.set_data_source()
        # Collect all the image names and corresponding label
        self.images = []
        self.get_image_names()  # [(image1,0),(image200,1)]

        # Transform

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

        image_name_label = self.images[index]
        image_label = image_name_label[1]
        image_ndarray = self.data_source(image_name_label)  # Already converted using cv2.imread

        if self.transform is not None:
            image_ndarray = self.tansform(image_ndarray)

        return image_ndarray, image_label



    def get_image_names(self):
        """
        Create a image data
        :return: None
        """

        # Iterate over all the NFS paths/ all the prefixes.
        for data_dir in self.data_dir:
            print("INFO: Reading datadir/prefix - {}".format(data_dir))
            if self.storage_format == "s3" and not validate_s3_prefix(data_dir):
                continue
            for category in self.label:
                category_index = self.label.index(category)

                if self.storage_format == "fs":
                    category_path = data_dir + "/" + category  # <base_image_dir>/<category>
                    print("INFO: Creating dataset for directory: {}/".format(category_path))
                    for root_path, dirs, files in os.walk(category_path, topdown=True):
                        for file in files:
                            file_path = root_path + "/" + file
                            self.images.append((file_path, category_index))
                else:
                    prefix = "{}{}/".format(data_dir, category)
                    print("INFO: Creating dataset for the prefix: {}".format(prefix))
                    for object_keys in self.s3_client.listObjects(bucket=self.bucket, prefix=prefix):
                        for object_key in object_keys:
                            # print("Object Key:{}".format(object_key))
                            self.images.append((object_key, category_index))

    def read_file_system_data(self, image):
        """
        Read data from file system.
        :param image:
        :return:
        """
        image_path = image[0]  # (image1,0) => (<image_name>,<Category Index>)
        category = self.label[image[1]]  # Find out category
        #image_path = self.data_dir + "/" + category + "/" + image_name
        img_ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read using CV2
        img_ndarray = cv2.resize(img_ndarray, self.image_dimension)
        return img_ndarray

    def read_s3_object(self, image):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        :param object_key:
        :return:
        """
        object_key = image[0]
        image_buffer = self.s3_client.getObject(Bucket=self.bucket, Key=object_key)
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

        # File system access
        if self.storage_name == "dss":
            if self.storage_format == "s3":
                self.credentials = storage_config[self.storage_name]["credentials"]
                self.data_dir = storage_config[self.storage_name]["prefix"]
                #if not self.data_dir.endswith("/"):
                #    self.data_dir += "/"
                self.bucket = storage_config[self.storage_name]["bucket"]
                self.s3_client = S3({"name": self.storage_name, "credentials": self.credentials})
                self.data_source = self.read_s3_object
        elif self.storage_name == "aws":
            self.credentials = storage_config[self.storage_name]["credentials"]
            if self.storage_format == "s3":
                pass
            elif self.storage_format == "fs":
                pass
            else:
                print("INFO: Wrong format ")
        elif self.storage_name in ["nfs", "ramfs"]:
            self.data_dir = storage_config[self.storage_name]["data_dir"]
            self.data_source = self.read_file_system_data
        else:
            print("ERROR: Wrong storage name! {}".format(self.storage_name))

        print("INFO: Data source:{}, format:{}".format(self.storage_name, self.storage_format))



class TorchImageClassificationDataset(RandomAccessDataset):
    """
    User defined dataset class. The base class has all default functionalities
    Each functions in the override section can be overide
    """

    def __init__(self, config={}):
        super(TorchImageClassificationDataset, self).__init__(
                                                              transform=None,
                                                              config=config)



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
    #def read_s3_object(self, image):
        """
        User need to update
        :param image: 
        :return: 
        """
    #    pass



class SequentialAccessDataset(Dataset):
    """
    This is Python iterator based
    """
    def __init__(self,config):
        self.config = config

    def __iter__(self):
        pass



class CustomDataset(object):
    def __init__(self,config):
        self.config = config
        self.random_access_dataset = config["dataset"]["access"]["random"]
        self.name = config["dataset"]["name"]
        self.class_name = self.get_class_name()
        self.dataset = None


    def get_class_name(self):
        """
        Convert the string class name to actual class
        :return:
        """
        try:
            return eval(self.name) # Convert the string to class name.
        except NameError as e:
            print("ERROR: Custom dataset doesn't exist! {}".format(e))
            sys.exit()

    def get_dataset(self):
        print("INFO: Using custom dataset - {}->{}".format(self.name,self.class_name))
        if self.random_access_dataset :
            return self.class_name(self.config)
        else:
            # This section should be for sequential read
            pass
