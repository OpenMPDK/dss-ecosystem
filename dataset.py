import os
import tensorflow as tf
import cv2
import random
from datetime import datetime





class DataSet(object):
    def __init__(self, config):
        print("Config:{}".format(config))
        self.data_dir = config.get("data_dir")
        self.label = config.get("label", [])
        self.dataset = []
        self.image_dimension = config.get("image_dimension",[40,40]) # height, width of image

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

    def create(self):
        """
        Create custom dataset for AI framework.
        [[<img-ndarray>,label1], [<img-ndarray>,label2], ... , [<img-ndarray>,labelN]]
        :return: None
        """
        start_time  = datetime.now()
        image_count = 0
        for category in self.label:
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

        print("INFO: dataset is initialized! Images:{}, Time: {} seconds".format(image_count, (datetime.now() - start_time).seconds))

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
