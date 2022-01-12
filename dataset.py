import os
import tensorflow as tf
import cv2
import random
from datetime import datetime
from s3_client import S3
import numpy as np




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


    def set_data_source(self):
        """
        Set the data source from configuration.
        :return: A function pointer
        """
        storage_config =  self.config["storage"]
        storage_type = storage_config["type"].lower()
        self.storage_format = storage_config["format"].lower()
        self.storage_name = storage_config["name"].lower()

        # File system access
        if storage_type == "local":
            if self.storage_format == "s3":
                self.credentials = storage_config[storage_type][self.storage_name]["credentials"]
                self.data_source = self.read_s3_object
            elif self.storage_format == "fs":
                self.data_dir = storage_config[storage_type][ self.storage_format]["data_dir"]
                self.data_source = self.read_file_system_data
            else:
                print("INFO: Wrong format ")

        elif storage_type == "cloud":
            self.credentials = storage_config[storage_type][self.storage_name]["credentials"]
            if self.storage_format == "s3":
                pass
            elif self.storage_format == "fs":
                pass
            else:
                print("INFO: Wrong format ")

        print("INFO: Data source:{}, format:{}".format(self.storage_name,self.storage_format))



    def create(self):
        """
        Create custom dataset for AI framework.
        [[<img-ndarray>,label1], [<img-ndarray>,label2], ... , [<img-ndarray>,labelN]]
        :return: None
        """
        start_time  = datetime.now()
        image_count = 0
        credential = {"endpoint": "http://10.1.51.2:9000", "access_key": "minio", "secret_key": "minio123"}
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
