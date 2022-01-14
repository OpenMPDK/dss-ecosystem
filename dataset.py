import os
import tensorflow as tf
import cv2
import random
import numpy as np
from datetime import datetime
from s3_client import S3
from utils.utility import exception

# PyTorch libraries
import torch
from torch.utils.data import Dataset



class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.config_dataset = config["dataset"]
        self.label = self.config_dataset["label"]
        self.dataset = []
        self.image_dimension = self.config_dataset["image_dimension"] # height, width of image
        self.data_source = None # Function to read data from
        self.credentials  = None # Required to access data from storage.
        self.storage_name = None # Storage name such as aws,dss
        self.storage_format = None # Data storage format such as s3, file system.
        self.data_dir = None

        # Call initial data-source setup functions
        self.set_data_source()

        # Create initial label for classification problem.
        self.create_label()

    def create_label(self):
        """
        Create label based on the category
        <data>/<dogs>
              /<cats>
              /<deers>
        label = ["dogs","cats","deers"]
        :return:
        """
        if not self.label :
            print("Annotation -  DataDir- {}".format(self.data_dir))
            for dir in os.listdir(self.data_dir):
                print("Label:{}".format(dir))
                self.label.append(dir)

    @exception
    def set_data_source(self):
        """
        Set the data source from configuration.
        :return: A function pointer
        """
        storage_config =  self.config["storage"]
        self.storage_format = storage_config["format"].lower()
        self.storage_name = storage_config["name"].lower()

        # File system access
        if self.storage_name == "dss":
            if self.storage_format == "s3":
                self.credentials = storage_config[self.storage_name]["credentials"]
                self.data_source = self.read_s3_object
        elif self.storage_name == "aws":
            self.credentials = storage_config[self.storage_name]["credentials"]
            if self.storage_format == "s3":
                pass
            elif self.storage_format == "fs":
                pass
            else:
                print("INFO: Wrong format ")
        elif self.storage_name in ["nfs","ramfs"]:
            self.data_dir = storage_config[self.storage_format]["data_dir"]
            self.data_source = self.read_file_system_data
        else:
            print("ERROR: Wrong storage name! {}".format(self.storage_name))

        print("INFO: Data source:{}, format:{}".format(self.storage_name,self.storage_format))



    def create(self):
        """
        Create custom dataset for AI framework.
        [[<img-ndarray>,label1], [<img-ndarray>,label2], ... , [<img-ndarray>,labelN]]
        :return: None
        """
        start_time  = datetime.now()
        image_count = 0
        for category in self.label:
            category_image_count = self.data_source(category)
            image_count += category_image_count
            #self.data_source(category)


        print("INFO: dataset is initialized! Images:{}, Time: {} seconds".format(image_count, (datetime.now() - start_time).seconds))

    def read_file_system_data(self,category):

        category_index = self.label.index(category)
        # Read file from category directory
        category_path = os.path.abspath(self.data_dir + "/" + category)

        for root_path, dirs, files in os.walk(category_path, topdown=True):
            for file in files:
                file_path = root_path + "/" + file
                img_ndarray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img_ndarray = cv2.resize(img_ndarray, self.image_dimension)  # Lets create all images in same shape.
                self.dataset.append([img_ndarray, category_index])
        category_image_count = len(files)
        print("INFO: Created dataset for category \"{}\" - Images: {} ".format(category, category_image_count))
        return category_image_count

    def read_s3_object(self, category):
        """

        :param category:
        :return:
        """
        # Create category index
        category_index = self.label.index(category)
        # Create S3 client with required parameters.
        s3 = S3({"name": self.storage_name, "credentials": self.credentials})
        object_count = 0
        for object_keys in s3.listObjects(bucket="bucket", prefix="flower_photos/{}/".format(category)):
            for object_key in object_keys:
                image_buffer = s3.getObject(Bucket="bucket",Key=object_key)
                image_numpy_array = np.asarray(bytearray(image_buffer), dtype="uint8")
                # Converts to image format
                image_2darray = cv2.imdecode(image_numpy_array, cv2.IMREAD_GRAYSCALE)
                image_2darray = cv2.resize(image_2darray, self.image_dimension)
                self.dataset.append([image_2darray, category_index ])
                object_count +=1
        print("INFO: Created dataset for category \"{}\" - Images: {} ".format(category,object_count))
        return object_count

    def get_dataset(self):
        return self.dataset

    def shuffle(self):
        """
        Shuffle dataset to have better accuracy.
        :return: None
        """
        if self.dataset:
            start_time = datetime.now()
            random.shuffle(self.dataset)
            print("INFO: data shuffling Time: {} seconds".format((datetime.now() - start_time).seconds))


    def test_dataset(self):
        base_image_dir = "/home/somnath.s/.keras/datasets/flower_photos/"
        # /home/somnath.s/.keras/datasets/flower_photos/roses/3475572132_01ae28e834_n.jpg
        # /home/somnath.s/.keras/datasets/flower_photos/sunflowers/5037531593_e2daf4c7f1.jpg
        # /home/somnath.s/.keras/datasets/flower_photos/tulips/489506904_9b68ba211c.jpg
        # /home/somnath.s/.keras/datasets/flower_photos/dandelion/4944731313_023a0508fd_n.jpg
        # /home/somnath.s/.keras/datasets/flower_photos/daisy/4897587985_f9293ea1ed.jpg

        test_image_set = [
            "3475572132_01ae28e834_n.jpg",
            "5037531593_e2daf4c7f1.jpg",
            "489506904_9b68ba211c.jpg",
            "4944731313_023a0508fd_n.jpg",
            "4897587985_f9293ea1ed.jpg"
        ]
        category_label = ["roses", "sunflowers", "tulips", "dandelion", "daisy"]

        dataset = []
        print("Test Image DImensions:{}".format(self.image_dimension))
        for index in range(len(test_image_set)):
            image = base_image_dir + "/" + category_label[index] + "/" + test_image_set[index]
            img_ndarray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            img_ndarray = cv2.resize(img_ndarray, self.image_dimension)  # Lets create all images in same shape.
            dataset.append(img_ndarray)

        return dataset



class TorchImageClassificationDataset(Dataset):
    """
    The CustomDataset inherits torch abstract class and override __len__ , __getitem__ functions.

    return size of the dataset and indexing
    """

    def __init__(self, transform=True, config={}):

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
        image_label =  image_name_label[1]
        image_ndarray = self.data_source(image_name_label) #  Already converted using cv2.imread
        image_ndarray = cv2.resize(image_ndarray, self.image_dimension)
        #if self.transform:
        #    image_ndarray = self.tansformation(image_ndarray)

        return image_ndarray, image_label


    def transformation(self, image_ndarray):
        image =  cv2.resize(image_ndarray, self.image_dimension)
        return image

    def get_image_names(self):
        """
        Create a image data
        :return: None
        """
        for category in self.label:
            category_index = self.label.index(category)
            category_path = self.data_dir + "/" + category   # <base_image_dir>/<category>
            print("INFO: Creating dataset for {}{}/".format(self.data_dir, category))
            if self.storage_format == "fs":
                for root_path, dirs, files in os.walk(category_path, topdown=True):
                    for file in files:
                        self.images.append((file, category_index))
            else:
                prefix = "{}{}/".format(self.data_dir,category)
                for object_keys in self.s3_client.listObjects(bucket=self.bucket, prefix=prefix):
                    for object_key in object_keys:
                        #print("Object Key:{}".format(object_key))
                        self.images.append((object_key,category_index))

    def read_file_system_data(self, image):
        image_name = image[0]  # (image1,0) => (<image_name>,<Category Index>)
        category   = self.label[image[1]] # Find out category
        image_path = self.data_dir + "/" + category + "/" + image_name
        img_ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read using CV2

        return img_ndarray

    def read_s3_object(self,image):
        """
        Read object from S3 , Any S3 compatible storage DSS, AWS-S3
        :param object_key:
        :return:
        """
        object_key = image[0]
        image_buffer = self.s3_client.getObject(Bucket=self.bucket, Key=object_key)
        image_numpy_array = np.asarray(bytearray(image_buffer), dtype="uint8")
        # Converts to image format
        image_2darray = cv2.imdecode(image_numpy_array, cv2.IMREAD_GRAYSCALE)
        return image_2darray

    @exception
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
                if not self.data_dir.endswith("/"):
                    self.data_dir += "/"
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
            self.data_dir = storage_config[self.storage_format]["data_dir"]
            self.data_source = self.read_file_system_data
        else:
            print("ERROR: Wrong storage name! {}".format(self.storage_name))

        print("INFO: Data source:{}, format:{}".format(self.storage_name, self.storage_format))