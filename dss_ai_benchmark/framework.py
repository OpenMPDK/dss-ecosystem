from worker import Worker

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# PyTorch library
import torch
from torchvision.models import resnet50
from torch.utils.data import DataLoader

# from dataset import DataSet, TorchImageClassificationDataset
from dataset import pytorch_dataset
from datetime import datetime
import numpy as np

# from models import NeuralNetwork, Net
from models import pytorch
from training import CustomTrain

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# PyTorch library
from torch.utils.data import DataLoader
import torch
import re
import collections
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')


class DNNFramework(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.features = []
        self.labels = []
        self.dataset = None

        self.framework = config.get("framework", {})
        self.name = self.framework.get("name", None)
        self.categories = self.config["dataset"][config["dataset"]["choice"]]["label"]

        # DNN Parameters
        self.epochs = self.framework["epochs"]
        self.batch_size = self.framework["batch_size"]
        self.max_batch_size = self.framework["max_batch_size"]

        self.model = None
        self.model_name = self.config["model"]["choice"]
        self.image_dimension = self.config["dataset"][config["dataset"]["choice"]]["image_dimension"]

        # Computation
        self.device = config["device"].lower()  # GPU /CPU
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
            self.train_dataset = self.dataset.dataset

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
        self.metrics[0] = ["dl_workers", "listing_workers", "listing_time (Sec)", "max_batch_size", "batch_size"]
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

    def __init__(self, config):
        import tensorflow as tf
        DNNFramework.__init__(self, config)
        self.logger.info("Running TensorFlow v{}".format(tf.__version__))

    def initialize(self):
        """
        Initialize the framework with creating dataset, dnn model and compiling a model.
        :return: None
        """
        self.create_dataset()
        self.divide_feature_label()

        # self.create_model()

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
        result = np.argmax(self.model.predict(test_dataset), axis=-1)
        self.logger.info("Predicted result - {}".format(result))
        result = self.model.predict_classes(test_dataset)
        self.logger.info("Predicted result - {}".format(result))


class PyTorch(DNNFramework):

    def __init__(self, config, logger):
        import torch
        from torchvision import transforms

        DNNFramework.__init__(self, config, logger)
        self.data_loader_params = self.framework["PyTorch"]["DataLoader"]
        self.distributed_data_parallel = self.framework["PyTorch"]["distributed_data_parallel"]
        self.logger.info("Using PyTorch v{}".format(torch.__version__))
        self.train_dataloader = None

        if "mean" in self.config["dataset"][config["dataset"]["choice"]]:
            self.config_mean = self.config["dataset"][config["dataset"]["choice"]]["mean"]
            self.config_std = self.config["dataset"][config["dataset"]["choice"]]["std"]
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config_mean, std=self.config_std)
            ])
        else:
            self.transforms = None

        self.default_collate_err_msg_format = (
            "default_collate: batch must contain tensors, numpy arrays, numbers, "
            "dicts or lists; found {}")

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
        custom_dataset = pytorch_dataset.CustomDataset(transforms=self.transforms, config=self.config,
                                                       logger=self.logger)
        self.dataset = custom_dataset.get_dataset()
        self.logger.info(f"Custom Dataset initialized, length: {len(self.dataset)}....")

    def create_data_loader(self):
        self.train_dataloader = DataLoader(self.dataset,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_batch,
                                           shuffle=self.data_loader_params["shuffle"],
                                           prefetch_factor=self.data_loader_params["prefetch_factor"],
                                           persistent_workers=self.data_loader_params["persistent_workers"],
                                           num_workers=self.data_loader_params["num_workers"],
                                           pin_memory=self.data_loader_params["pin_memory"],
                                           drop_last=self.data_loader_params["drop_last"]
                                           )

        self.logger.info("DataLoader initialized with workers:{}, max_batch_size:{}, batch_size:{}".format(
            self.data_loader_params["num_workers"], self.max_batch_size, self.batch_size))

    def collate_batch(self, batch):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(self.default_collate_err_msg_format.format(elem.dtype))

                return self.collate_batch([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.collate_batch([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.collate_batch(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.collate_batch(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def create_model(self):
        """
        Create NeuralNetwork model
        :return:
        """
        self.logger.info("Creating AI model! - device:{}".format(self.device))

        torch_model = pytorch.Model(name=self.model_name,
                                    num_classes=len(self.categories),
                                    image_dimension=self.image_dimension,
                                    device=self.device,
                                    logger=self.logger)

        self.model = torch_model.get()

        # self.model = pytorch.Model(self.image_dimension).to(self.device)

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
        if self.config["model"]["choice"] == "ObjectDetector":
            self.logger.info(f'\n\nPLEASE NOTE: The corresponding Inference to this Object Detection framework'
                             f' is NOT a part of this Benchmark script; a separate script has been created for the same'
                             f': **object_detector_predict.py** (Present in this same project folder).'
                             f'\n\n----------Run Instructions----------\n'
                             f'The above script can be run with test images present on Filesystem or S3.\n\n'
                             f'To run with test images from Filesystem:\n'
                             f'$ python3 object_detector_predict.py --fs --input absolute/path/to/the/test/image.jpg\n'
                             f'Or, for bulk image tests:\n'
                             f'$ python3 object_detector_predict.py --fs --input '
                             f'absolute/path/to/the/text/file/containing/a/list/of/images.txt '
                             f'\n\nTo run with test images from S3:\n'
                             f'$ python3 object_detector_predict.py --s3 --input client_lib_name(dss_client / '
                             f'boto3):prefix/to/the/test/image.jpg\n '
                             f'Or, for bulk image tests:\n'
                             f'$ python3 object_detector_predict.py --fs --input client_lib_name(dss_client / '
                             f'boto3):prefix/to/the/text/file/containing/a/list/of/images.txt\n'
                             f'**In case of bulk image tests with S3 inputs, please make sure that the "text" file'
                             f'contains the list of the test images along with its full prefix.\n\n')
        else:
            self.logger.info(f'\n\nPLEASE NOTE: The corresponding Inference to this Image Classification framework'
                             f' is NOT a part of this Benchmark script; a separate script has been created for the same'
                             f': **image_classifier_predict.py** (Present in this same project folder).'
                             f'\n\n----------Run Instructions----------\n'
                             f'The above script can be run with test images present on Filesystem or S3.\n\n'
                             f'To run with test images from Filesystem:\n'
                             f'$ python3 image_classifier_predict.py --fs --input absolute/path/to/the/test/image.jpg\n'
                             f'Or, for bulk image tests:\n'
                             f'$ python3 image_classifier_predict.py --fs --input '
                             f'absolute/path/to/the/text/file/containing/a/list/of/images.txt '
                             f'\n\nTo run with test images from S3:\n'
                             f'$ python3 image_classifier_predict.py --s3 --input client_lib_name(dss_client / '
                             f'boto3):prefix/to/the/test/image.jpg\n '
                             f'Or, for bulk image tests:\n'
                             f'$ python3 image_classifier_predict.py --fs --input client_lib_name(dss_client / '
                             f'boto3):prefix/to/the/text/file/containing/a/list/of/images.txt\n'
                             f'**In case of bulk image tests with S3 inputs, please make sure that the "text" file'
                             f'contains the list of the test images along with its full prefix.\n\n')
