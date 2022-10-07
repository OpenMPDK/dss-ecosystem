import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import torch.nn.functional as F

from abc import abstractmethod


class Model(object):
    """
    Create the model based on the specification from configuration file.
    DO NOT Touch
    """

    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.num_classes = kwargs["num_classes"]
        self.device = kwargs["device"]
        self.logger = kwargs["logger"]
        self.image_dimension = kwargs["image_dimension"]

        # Load the pretrained network for object detection
        self.baseModel = resnet50(pretrained=True)
        # freeze all ResNet50 layers, so they will *not* be updated during the training process
        for param in self.baseModel.parameters():
            param.requires_grad = False

        self.model_class_name = self.get_class_name()

    def get_class_name(self):
        try:
            return eval(self.name)
        except NameError as e:
            self.logger.fatal("{}".format(e))
        except Exception as e:
            self.logger.fatal("ERROR: {}".format(e))
        return None

    def get(self):
        self.logger.info(
            "INFO: Creating neural network model instance - {}=>{}".format(self.name, self.model_class_name))
        if self.model_class_name:
            if str(self.name) == 'ObjectDetector':
                return self.model_class_name(self.baseModel, self.num_classes).to(self.device)
            else:
                return self.model_class_name(self.image_dimension, self.logger).to(self.device)


class NeuralNetwork(nn.Module):
    """
    Need to create a base class following default functions
    Encouraged to override the function into child class.
    network()
    loss_function()
    forward()
    """

    def __init__(self, logger):
        self.logger = logger
        super(NeuralNetwork, self).__init__()

    def loss_function(self):
        return nn.CrossEntropyLoss()

    @abstractmethod
    def forward(self):
        self.logger.fatal("Should be implemented in child class!")

    @abstractmethod
    def network(self):
        self.logger.fatal("Should be implemented in child class!")


class SequentialNet(NeuralNetwork):

    def __init__(self, image_diments=(), logger=None):
        super(SequentialNet, self).__init__(logger=logger)
        self.input_image_height = int(image_diments[0])
        self.input_image_width = int(image_diments[1])
        self.logger = logger

        self.network()

    def network(self):
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_image_height * self.input_image_weidth, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ConvNet(NeuralNetwork):
    """
    Expected a image dimesion of 32x32 because of tensor size
    Train Data:torch.Size([64, 32, 32]),torch.Size([64])
    """
    def __init__(self, image_diments=(), logger=None):
        super(ConvNet, self).__init__(logger=logger)
        self.input_image_height = int(image_diments[0])
        self.input_image_width = int(image_diments[1])
        self.logger = logger

        self.network()

    def network(self):
        # <Input Channel>,<Output Channel>, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Feed 16 out channel from above convolution to linear layer with image dimensions 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)  # <input> ,<output>
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # Max pooling over a (2,2) window ( Sub sampling ) of the output from first feature maps.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ObjectDetector(nn.Module):

    def __init__(self, baseModel=None, num_classes=0):
        super(ObjectDetector, self).__init__()
        self.numClasses = num_classes
        self.baseModel = baseModel

        # build the regressor head for outputting the bounding box coordinates
        self.regressor = nn.Sequential(
            nn.Linear(baseModel.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        # build the classifier head to predict the class labels
        self.classifier = nn.Sequential(
            nn.Linear(baseModel.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.numClasses)
        )

        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network

        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)

        # return the outputs as a tuple
        return bboxes, classLogits
