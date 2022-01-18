
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# PyTorch library
from torch.utils.data import DataLoader
import torch.optim as optimization
from dataset import DataSet, TorchImageClassificationDataset
from datetime import datetime
import numpy as np

from models import NeuralNetwork, Net

class DNNFramework(object):
    def __init__(self, config):
        self.config = config
        self.name = config["framework"]["name"]
        self.features = []
        self.labels = []
        self.dataset = None

        self.framework = config.get("framework", {})
        self.framework_name = self.framework.get("name", None)
        self.categories = config["dataset"].get("label", [])

        # DNN Parameters
        self.framework_epochs = self.framework["epochs"]
        self.framework_batch_size = self.framework["batch_size"]

        self.model = None
        self.image_dimension = config["dataset"]["image_dimension"]

        # Computation
        self.device = config["device"].lower() # GPU /CPU
        if not self.device or self.device == "gpu":
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'


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
        # Load few required libraries
        #import torch.optim as optimization

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
        train_image , train_label = dataiter.next()
        print("Train Data:{},{}".format(train_image.size(),train_label.size()))
    def create_model(self):
        print("Device:{}".format(self.device))
        #self.model = NeuralNetwork().to(self.device)
        self.model = Net()
        print(self.model)
    def training(self):
        criterion = self.model.loss_function()
        optimizer = optimization.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.framework_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                images, labels = data
                # Zero the parameter gradient
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                loss = criterion(outputs,labels)  # loss calculation based on CrossEntropy
                loss.backward()
                optimizer.step()

                # Print train stats
                running_loss += loss.item()
                if i % 2000 ==  1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print("INFO: Training is done")



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
