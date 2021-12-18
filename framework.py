
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from dataset import DataSet
from datetime import datetime
import numpy as np

class DNNFramework(object):
    def __init__(self, config):
        self.config = config
        self.name = config["framework"]["name"]
        self.features = []
        self.labels = []
        self.dataset = None

        self.computation = config.get("computation", "CPU")
        self.framework = config.get("framework", {})
        self.framework_name = self.framework.get("name", None)
        self.categories = config["dataset"].get("label", [])

        # DNN Parameters
        self.framework_epochs = self.framework["epochs"]
        self.framework_batch_size = self.framework["batch_size"]

        self.model = None

    def create_dataset(self):
        if self.dataset is None:
            ds = DataSet(self.config.get("dataset",{}))
            ds.create()
            ds.shuffle()  # Shuffle the dataset to have better accuracy
            self.dataset =  ds.dataset

    def divide_feature_label(self):
        """
        Classification problem should have feature and corresponding label.
        A feature is a numpy list of numpy array of images
        Label is a numpy array of category index.
        :return:
        """
        for feature, label in self.dataset:
            self.features.append(feature)
            self.labels.append(label)
        # Convert np-array
        self.features = np.array(self.features).reshape(-1, 180, 180, 1)
        self.labels = np.array(self.labels)


class TensorFlow(DNNFramework):

    def __init__(self,config):
        DNNFramework.__init__(self,config)

    def initialize(self):
        """
        Initialize the framework with creating dataset, dnn model and compiling a model.
        :return: None
        """
        self.create_dataset()
        self.divide_feature_label()

        self.create_model()

    def create_model(self):
        """
        Create model
        #TODO support of different type of models.
        :return:
        """
        # Create model
        # Based on AI framework
        self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5)
        ])

        # Compile model
        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

    def training(self):
        """
        Training on CPU or single GPU
        :return:
        """
        # Train/fit model
        self.model.fit(self.features, self.labels, batch_size=self.framework_batch_size, epochs=self.framework_epochs)
        # Model
    def distributed_training(self):
        """
        Training across multiple GPU
        :return:
        """
        pass

    def inference(self):
        pass

class PyTorch(DNNFramework):

    def __init__(self,config):
        DNNFramework.__init__(self,config)

    def training(self):
        pass