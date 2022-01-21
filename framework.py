
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# PyTorch library
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optimization
from dataset import DataSet, TorchImageClassificationDataset
from datetime import datetime
import numpy as np

#from models import NeuralNetwork, Net
from models import pytorch

class DNNFramework(object):
    def __init__(self, config):
        self.config = config
        self.features = []
        self.labels = []
        self.dataset = None

        self.framework = config.get("framework", {})
        self.name = self.framework.get("name", None)
        self.categories = config["dataset"].get("label", [])

        # DNN Parameters
        self.epochs = self.framework["epochs"]
        self.batch_size = self.framework["batch_size"]

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
        print("INFO: Inference ...")
        test_dataset = self.dataset.test_dataset()
        result = np.argmax(self.model.predict(test_dataset) , axis=-1)
        print("INFO: Predicted result - {}".format(result))
        result  = self.model.predict_classes(test_dataset)
        print("INFO: Predicted result - {}".format(result))



class PyTorch(DNNFramework):

    def __init__(self,config):
        DNNFramework.__init__(self,config)
        self.data_loader_params = self.framework["PyTorch"]["DataLoader"]


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

        #self.create_model()


    def create_dataset(self):
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = TorchImageClassificationDataset(transform=transform,
                                                        config=self.config)

    def create_data_loader(self):
        self.train_dataloader = DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           shuffle = self.data_loader_params["shuffle"],
                                           prefetch_factor=self.data_loader_params["prefetch_factor"],
                                           persistent_workers=self.data_loader_params["persistent_workers"],
                                           pin_memory=self.data_loader_params["pin_memory"],
                                           drop_last=self.data_loader_params["drop_last"]
        )

        dataiter = iter(self.train_dataloader)
        train_image , train_label = dataiter.next()
        print("Train Data:{},{}".format(train_image.size(),train_label.size()))
    def create_model(self):
        """
        Create NeuralNetwork model
        :return:
        """
        print("INFO: Creating AI model! - device:{}".format(self.device))
        #self.model = NeuralNetwork(self.image_dimension).to(self.device)
        self.model = pytorch.CNN(self.image_dimension).to(self.device)
        #self.model = Net().to(self.device)
        print(self.model)
    def training(self):
        """
        Train a model
        :return:
        """
        start_time = datetime.now()
        print(f"INFO: Training started!{start_time}")
        criterion = self.model.loss_function()
        optimizer = optimization.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Following line returns image, label tensor.
            j = 0
            for batch_index, data in enumerate(self.train_dataloader, 0):
                images, labels = data
                images = images.float() # Convert to float.
                #print(images[0])
                # Zero the parameter gradient
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                loss = criterion(outputs,labels)  # loss calculation based on CrossEntropy
                loss.backward()
                optimizer.step()

                # Add loss for 10 batches.
                running_loss += loss.item()
                if batch_index % 10 ==  0:
                    #print("Batch Index:{}, ImageTensor:{}, LabelTensor:{}".format(batch_index, len(images), len(labels)))
                    print(f'Epoch:{epoch + 1}, BatchIndex:{batch_index} loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

        print("INFO: Training is done : {} seconds".format( (datetime.now() - start_time ).seconds))



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
