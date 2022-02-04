
from abc import abstractmethod, abstractproperty

# Torch Libraries
import torch.optim as optimization
import torchvision.transforms as transforms
import time

class DNNTrain(object):

    def __init__(self,**kwargs):
        self.config = kwargs["config"]
        self.model = kwargs["model"]
        self.device = kwargs["device"]
        self.logger = kwargs["logger"]
        # DNN Parameters
        self.framework = self.config["framework"]
        self.framework_name = self.framework["name"].lower()
        self.epochs = self.framework["epochs"]
        self.batch_size = self.framework["batch_size"]
        self.max_batch_size = self.framework["max_batch_size"]
        self.image_dimension = self.config["dataset"]["image_dimension"]

        self.load_libraries()

    def load_libraries(self):
        if self.framework_name == "pytorch":
            pass
        elif self.framework_name == "tensorflow":
            pass


    @abstractmethod
    def train(self):
        self.logger.error("Create your own train method!")

#### TensorFlow ####
#class TFTrain(DNNTrain):
#    def __init__(self,config):
#        super(TFTrain,self).__init__(config)

#    def train(self):
#        self.model.fit(self.features, self.labels, batch_size=self.batch_size, epochs=self.epochs)



#### PyTorch  ####
class RandomAccessDatasetTrain(DNNTrain):
    """
    Example of training for MapStyle dataset.
    """
    def __init__(self,**kwargs):
        self.config = kwargs["config"]
        self.train_dataloader = kwargs["dataloader"]
        super(RandomAccessDatasetTrain,self).__init__(config=self.config,
                                       model=kwargs["model"],
                                       device=kwargs["device"],
                                       logger=kwargs["logger"])

    def train(self):
        start_time = time.monotonic()
        self.logger.info(f"INFO: Training started! {start_time}")
        criterion = self.model.loss_function()
        optimizer = optimization.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Following line returns image, label tensor.
            j = 0
            for batch_index, data in enumerate(self.train_dataloader, 0):
                images, labels = data
                images = images.float()  # Convert to float.
                # Zero the parameter gradient
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(images)
                loss = criterion(outputs, labels)  # loss calculation based on CrossEntropy
                loss.backward()
                optimizer.step()

                # Add loss for 10 batches.
                running_loss += loss.item()
                if batch_index % self.max_batch_size == 0:
                    self.logger.info(f'Epoch:{epoch + 1}, BatchIndex:{batch_index} loss: {running_loss / self.max_batch_size:.3f}')
                    running_loss = 0.0
        total_time = time.monotonic() - start_time
        bw = (self.train_dataloader.dataset.dataset_size_in_bytes.value /1024) / total_time
        self.logger.info("Training is done : {:.2f} seconds, BW:{:.2f} MiB/Sec".format(total_time, bw))

class PythonReadTrain(DNNTrain):

    def __init__(self,**kwargs):
        self.config = kwargs["config"]
        self.train_dataloader = kwargs["dataloader"]
        super(PythonReadTrain, self).__init__(**kwargs)

    def train(self):
        start= time.monotonic()
        for epoch in range(self.epochs):
            for batch_index, data in enumerate(self.train_dataloader):
                if batch_index == self.max_batch_size - 1:
                    break
        total_time = time.monotonic() - start
        bw = (self.train_dataloader.dataset.dataset_size_in_bytes.value/1024) / total_time
        self.logger.info("1 epoch time: {:.2f} Sec, {} KBytes, BW:{:.2f} MiB/Sec".format(total_time, self.train_dataloader.dataset.dataset_size_in_bytes.value, bw))


class SequentialDatasetTrain(DNNTrain):
    """
    Example training for sequential dataset.
    #TODO
    """
    def __init__(self,config):
        super(SequentialDatasetTrain,self).__init__(config)

    def train(self):
        pass


class DDPTrain(DNNTrain):
    def __init__(self,config):
        super(DDPTrain,self).__init__(config)

    def train(self):
        pass


class CustomTrain(object):
    """
    DO NOT Touch
    """
    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.dataloader = kwargs["dataloader"]
        self.model = kwargs["model"]
        self.device = kwargs["device"]
        self.name = self.config["training"]["name"]
        self.logger = kwargs["logger"]
        self.class_name = self.get_class_name()

    def get_class_name(self):
        try:
            return eval(self.name)
        except NameError as e:
            self.logger.fatal("{}".format(e))
        except Exception as e:
            self.logger.fatal("{}".format(e))

    def start(self):
        try:
            custom_train = self.class_name(config=self.config,
                                           dataloader=self.dataloader,
                                           model=self.model,
                                           device=self.device,
                                           logger=self.logger
                                           )
            # custom_train.create_model()
            custom_train.train()  # Initiate training
        except Exception as e:
            self.logger.error("{}".format(e))






