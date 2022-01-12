import os
import tensorflow as tf
import cv2
import random
from datetime import datetime
from s3_client import S3





class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.config_dataset = config["dataset"]
        self.data_dir = self.config_dataset["data_dir"]
        self.label = self.config_dataset["label"]
        self.dataset = []
        self.image_dimension = self.config_dataset["image_dimension"] # height, width of image

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
        storage_format = storage_config["format"].lower()
        storage_name = storage_config["name"].lower()

        # File system access
        if storage_type == "local":
            if storage_format == "s3":
                credentials = storage_config[storage_name]["credentials"]
                return self.read_s3_object(storage_name,credentials)
            elif storage_format == "fs":
                pass
            else:
                print("INFO: Wrong format ")

        elif storage_type == "cloud":
            credentials = storage_config[storage_name]["credentials"]
            if storage_format == "s3":
                pass
            elif storage_format == "fs":
                pass
            else:
                print("INFO: Wrong format ")



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
            category_image_count = self.read_file_system_data(category)
            image_count += category_image_count
            #self.read_s3_object("dss", credential, category)
            """
            category_index = self.label.index(category)
            # Read file from category directory
            category_path = os.path.abspath(self.data_dir + "/" + category)

            for root_path, dirs, files in os.walk(category_path, topdown=True):
                for file in files:
                    file_path = root_path + "/" + file
                    img_ndarray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    img_ndarray = cv2.resize(img_ndarray, self.image_dimension) # Lets create all images in same shape.
                    self.dataset.append([img_ndarray, category_index])
            category_image_count = len(files)
            print("INFO: Created dataset for category \"{}\" - Images: {} ".format(category, category_image_count ))
            image_count +=category_image_count
            """

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

    def read_s3_object(self, storage_name, credentials, category):
        params = {"name": storage_name, "credentials": credentials}
        s3 = S3(params)
        for objects in s3.listObjects(bucket="bucket", prefix="flower_photos/{}/".format(category)):
            print("Total Objects Listed: {}".format(len(objects)))
            print(objects[0], objects[-1])

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
