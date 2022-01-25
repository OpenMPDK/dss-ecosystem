
from abc import abstractmethod, abstractproperty
from datetime import datetime

# Torch Libraries
import torch.optim as optimization
import torchvision.transforms as transforms

class DNNTrain(object):

    def __init__(self,**kwargs):
        self.config = kwargs["config"]
        self.model = kwargs["model"]
        self.device = kwargs["device"]
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
        print("ERROR:Create your own train method!")

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
                                       device=kwargs["device"])

    def train(self):
        start_time = datetime.now()
        print(f"INFO: Training started! {start_time}")
        criterion = self.model.loss_function()
        optimizer = optimization.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Following line returns image, label tensor.
            j = 0
            for batch_index, data in enumerate(self.train_dataloader, 0):
                images, labels = data
                images = images.float()  # Convert to float.
                # print(images[0])
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
                    # print("Batch Index:{}, ImageTensor:{}, LabelTensor:{}".format(batch_index, len(images), len(labels)))
                    print(f'Epoch:{epoch + 1}, BatchIndex:{batch_index} loss: {running_loss / self.max_batch_size:.3f}')
                    running_loss = 0.0

        print("INFO: Training is done : {} seconds".format((datetime.now() - start_time).seconds))

class PythonReadTrain(DNNTrain):

    def __init__(self,**kwargs):
        self.config = kwargs["config"]
        self.train_dataloader = kwargs["dataloader"]
        super(PythonReadTrain, self).__init__(**kwargs)

    def train(self):
        start= time.time()
        for epoch in range(self.epochs):
            for batch_index, data in enumerate(self.train_dataloader):
                if batch_index == self.max_batch_size - 1:
                    total_time = time.time() - start
                    print("1 epoch time: {}".format(total_time))
                    break

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
        self.class_name = self.get_class_name()

    def get_class_name(self):
        try:
            return eval(self.name)
        except NameError as e:
            print("ERROR: {}".format(e))
        except Exception as e:
            print("ERROR: {}".format(e))

    def start(self):
        try:
            s = DNNTrain(config=self.config,
                         model=self.model,
                         device=self.device)
        except Exception as e:
            print("TTTT:{}".format(e))
        custom_train = self.class_name(config=self.config,
                                       dataloader=self.dataloader,
                                       model=self.model,
                                       device=self.device
                                      )
        #custom_train.create_model()
        custom_train.train() # Initiate training





