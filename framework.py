
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# PyTorch library
from torch.utils.data import DataLoader
from dataset import DataSet, TorchImageClassificationDataset
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
        self.image_dimension = config["dataset"]["image_dimension"]

        # Computation
        self.computation = config["computation"]

        # Execution
        if config["execution"]["sequential"]:
            self.execution =  "sequential"
        else:
            self.execution = "parallel"

    def create_dataset(self):
        if self.dataset is None:
            self.dataset = DataSet(self.config)
            self.dataset.create()
            self.dataset.shuffle()  # Shuffle the dataset to have better accuracy
            self.train_dataset =  self.dataset.dataset

    def divide_feature_label(self):
        """
        Classification problem should have feature and corresponding label.
        A feature is a numpy list of numpy array of images
        Label is a numpy array of category index.
        :return:
        """
        for feature, label in self.train_dataset:
            self.features.append(feature)
            self.labels.append(label)
        print("INFO:Features Size:{}, Labels:{}".format(len(self.features), len(self.labels)))
        # Convert np-array
        self.features = np.array(self.features).reshape(-1, self.image_dimension[0], self.image_dimension[1], 1)
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
        """
        The inference process predicts category for a random image.
        :return:
        """
        print("INFO: Inference ...")
        test_dataset = self.dataset.test_dataset()
        result = np.argmax(self.model.predict(test_dataset) , axis=-1)
        print("INFO: Predicted result - {}".format(result))
        result  = self.model.predict_classes(test_dataset)
        print("INFO: Predicted result - {}".format(result))



class PyTorch(DNNFramework):

    def __init__(self,config):
        DNNFramework.__init__(self,config)

    def initialize(self):
        """
        Initialize the framework with creating dataset, dnn model and compiling a model.
        :return: None
        """
        self.create_dataset()
        self.create_data_loader()
        #self.divide_feature_label()

        self.create_model()


    def create_dataset(self):
        self.dataset = TorchImageClassificationDataset(transform=True,
                                                        config=self.config)
        #print(self.dataset[0])

    def create_data_loader(self):
        self.train_dataloader = DataLoader(self.dataset,
                                           batch_size=self.framework_batch_size,
                                           shuffle = True
        )

        dataiter = iter(self.train_dataloader)
        image , label = dataiter.next()
        print("Train Data:{},{}".format(image,label))
    def create_model(self):
        pass

    def training(self):
        pass

    def training_distributed_data_parallel(self):
        """
        Works only on GPU
        :return:
        """
        pass

    def training_map_style(self):
        """
        Works both on CPU/GPU
        :return:
        """
        pass

    def inference(self):
        """
        Predict the category of an unknown image.
        :return:
        """
        pass
