

from worker import Worker


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# PyTorch library
from torch.utils.data import DataLoader


#from dataset import DataSet, TorchImageClassificationDataset
from dataset import pytorch_dataset
from datetime import datetime
import numpy as np

#from models import NeuralNetwork, Net
from models import pytorch
from training import CustomTrain

class DNNFramework(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.features = []
        self.labels = []
        self.dataset = None

        self.framework = config.get("framework", {})
        self.name = self.framework.get("name", None)
        self.categories = config["dataset"]["label"]

        # DNN Parameters
        self.epochs = self.framework["epochs"]
        self.batch_size = self.framework["batch_size"]
        self.max_batch_size = self.framework["max_batch_size"]

        self.model = None
        self.model_name = self.config["model"]["name"]
        self.image_dimension = config["dataset"]["image_dimension"]

        # Computation
        self.device = config["device"].lower() # GPU /CPU
        if not self.device or self.device == "gpu":
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        # Workers:
        self.listing_workers = config["execution"]["workers"]
        # Metrics
        self.metrics = []
        self.metrics_train = []


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
        self.logger.info("Features Size:{}, Labels:{}".format(len(self.features), len(self.labels)))
        # Convert np-array
        self.features = np.array(self.features).reshape(-1, self.image_dimension[0], self.image_dimension[1], 1)
        self.labels = np.array(self.labels)

    def update_metrics(self):

        for index in range(2):
            self.metrics.append([])
        # Update metrics
        self.metrics[0] = ["dl_workers","listing_workers","listing_time", "max_batch_size","batch_size"]
        self.metrics[1] = [str(self.dataset.data_loader_workers),
                           str(self.dataset.max_workers),
                           str(self.dataset.listing_time),
                           str(self.max_batch_size),
                           str(self.batch_size)
                           ]

        # Update train metrics
        if self.metrics_train:
            self.metrics[0].extend(self.metrics_train[0])
            self.metrics[1].extend(self.metrics_train[1])




class TensorFlow(DNNFramework):

    def __init__(self,config):
        import tensorflow as tf
        DNNFramework.__init__(self,config)
        self.logger.info("Running TensorFlow v{}".format(tf.__version__))

    def initialize(self):
        """
        Initialize the framework with creating dataset, dnn model and compiling a model.
        :return: None
        """
        self.create_dataset()
        self.divide_feature_label()

        #self.create_model()

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
        self.model.fit(self.features, self.labels, batch_size=self.batch_size, epochs=self.epochs)
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
        self.logger.info("Inference ...")
        test_dataset = self.dataset.test_dataset()
        result = np.argmax(self.model.predict(test_dataset) , axis=-1)
        self.logger.info("Predicted result - {}".format(result))
        result  = self.model.predict_classes(test_dataset)
        self.logger.info("Predicted result - {}".format(result))



class PyTorch(DNNFramework):

    def __init__(self,config,logger):
        import torch
        DNNFramework.__init__(self,config, logger)
        self.data_loader_params = self.framework["PyTorch"]["DataLoader"]
        self.distributed_data_parallel =  self.framework["PyTorch"]["distributed_data_parallel"]
        self.logger.info("Using PyTorch v{}".format(torch.__version__))


    def initialize(self):
        """
        Initialize the framework with creating dataset, dnn model and compiling a model.
        :return: None
        """
        # Load few required libraries
        self.create_dataset()
        self.create_data_loader()

    def create_dataset(self):
        """
        Create custom dataset.
        :return:
        """
        custom_dataset = pytorch_dataset.CustomDataset(config=self.config, logger=self.logger)
        self.dataset = custom_dataset.get_dataset()

    def create_data_loader(self):
        self.train_dataloader = DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           shuffle = self.data_loader_params["shuffle"],
                                           prefetch_factor=self.data_loader_params["prefetch_factor"],
                                           persistent_workers=self.data_loader_params["persistent_workers"],
                                           num_workers=self.data_loader_params["num_workers"],
                                           pin_memory=self.data_loader_params["pin_memory"],
                                           drop_last=self.data_loader_params["drop_last"]
        )

        self.logger.info("DataLoader initialized with workers:{}, max_batch_size:{}, batch_size:{}".format(
            self.data_loader_params["num_workers"], self.max_batch_size, self.batch_size))
    def create_model(self):
        """
        Create NeuralNetwork model
        :return:
        """
        self.logger.info("Creating AI model! - device:{}".format(self.device))
        torch_model = pytorch.Model(name=self.model_name,
                                    image_dimension=self.image_dimension,
                                    device=self.device,
                                    logger=self.logger)
        self.model = torch_model.get()

        #self.model = pytorch.Model(self.image_dimension).to(self.device)

        self.logger.info("{}".format(self.model))

    def training(self):
        """
        Train a model
        :return:
        """
        train = CustomTrain(config=self.config,
                            dataloader=self.train_dataloader,
                            model=self.model,
                            device=self.device,
                            metrics=self.metrics_train,
                            logger=self.logger)
        train.start()

    def inference(self):
        """
        Predict the category of an unknown image.
        :return:
        """
        pass
