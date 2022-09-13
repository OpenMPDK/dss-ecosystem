import os
from abc import abstractmethod, abstractproperty
import numpy as np

# Torch Libraries
import torch
import torch.optim as optimization
import torchvision.transforms as transforms

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss

import matplotlib.pyplot as plt

from tqdm import tqdm

import time


class DNNTrain(object):

    def __init__(self, **kwargs):
        self.config = kwargs["config"]
        self.model = kwargs["model"]
        self.device = kwargs["device"]
        self.logger = kwargs["logger"]
        self.train_dataloader = kwargs["dataloader"]
        # DNN Parameters
        self.framework = self.config["framework"]
        self.framework_name = self.framework["name"].lower()
        self.epochs = self.framework["epochs"]
        self.batch_size = self.framework["batch_size"]
        self.max_batch_size = self.framework["max_batch_size"]
        self.image_dimension = self.config["dataset"][self.config["dataset"]["choice"]]["image_dimension"]
        self.num_workers = self.framework[self.framework["name"]]["DataLoader"]["num_workers"]

        # Metrics
        self.metrics = kwargs["metrics"]

        self.load_libraries()

    def load_libraries(self):
        if self.framework_name == "pytorch":
            pass
        elif self.framework_name == "tensorflow":
            pass

    @abstractmethod
    def train(self):
        self.logger.error("Create your own train method!")

# ### TensorFlow ####
# class TFTrain(DNNTrain):
#     def __init__(self,config):
#         super(TFTrain,self).__init__(config)

#     def train(self):
#         self.model.fit(self.features, self.labels, batch_size=self.batch_size, epochs=self.epochs)


#### PyTorch  ####
class RandomAccessDatasetTrain(DNNTrain):
    """
    Example of training for MapStyle dataset.
    """
    def __init__(self, **kwargs):
        super(RandomAccessDatasetTrain, self).__init__(**kwargs)

    def train(self):
        """
        Train the model with valid dataset.
        :return: None
        """
        self.logger.info("INFO: Training started!")
        criterion = self.model.loss_function()
        optimizer = optimization.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        start_time = time.monotonic()
        # Add metrics header
        self.metrics.append(
            ["num_epochs", "total_time (Sec)", "dataset_size (MBytes)", "read_bw (MB/Sec)", "dataset_read_time (Sec)"])
        tot_read_time, dataload_time, epoch_bw = [], [], []
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Following line returns image, label tensor.
            self.train_dataloader.dataset.dataset_size_in_bytes.value = 0
            epoch_start_time = time.monotonic()
            epoch_read_time = 0
            dload_elapsed = 0
            for batch_index, data in enumerate(self.train_dataloader, 0):
                images, labels, read_times = data
                images = images.float()  # Convert to float.

                epoch_read_time += torch.mean(read_times).item()

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
                    self.logger.info(
                        f'Epoch:{epoch + 1}, BatchIndex:{batch_index} loss: {running_loss / self.max_batch_size:.3f}')
                    running_loss = 0.0

                dload_elapsed += (time.monotonic() - epoch_start_time)
                epoch_start_time = time.monotonic()

            tot_read_time.append(round(epoch_read_time, 4))
            dataload_time.append(round(dload_elapsed, 4))
            dataset_size_mb = round((self.train_dataloader.dataset.dataset_size_in_bytes.value / 1024), 2)
            epoch_bw.append(round((dataset_size_mb / epoch_read_time), 3))
        self.metrics.append([str(self.epochs).replace(',', ';'), str(dataload_time).replace(',', ';'),
                             str(dataset_size_mb).replace(',', ';'),
                             str(epoch_bw).replace(',', ';'), str(tot_read_time).replace(',', ';')])
        total_time = round(time.monotonic() - start_time, 4)
        bw = [round((self.train_dataloader.dataset.dataset_size_in_bytes.value / 1024) / t, 3) for t in tot_read_time]
        train_summary = "** Train Summary **\n"
        train_summary += "\t Epochs:{}, BatchSize:{}, MaxBatchSize:{}\n".format(self.epochs, self.batch_size,
                                                                                self.max_batch_size)
        train_summary += "\t Loading+Training Time:{:.2f} Sec, Dataset Size:{} KBytes, Read_BW:{} MiB/Sec, " \
                         "Total Read Time:{} Secs\n".format(total_time,
                                                            self.train_dataloader.dataset.dataset_size_in_bytes.value,
                                                            bw, tot_read_time)
        self.logger.info(train_summary)


class ObjectDetectionDatasetTrain(DNNTrain):

    def __init(self, **kwargs):
        super(ObjectDetectionDatasetTrain, self).__init__(**kwargs)

    def train(self):
        # define our loss functions
        classLossFunc = CrossEntropyLoss()
        bboxLossFunc = MSELoss()

        steps_per_epoch = len(self.train_dataloader.dataset) // self.config["framework"]["batch_size"]

        # initialize the optimizer, compile the model, and show the model summary
        opt = Adam(self.model.parameters(),
                   lr=self.config["training"][self.config["training"]["choice"]]["init_lr"])

        # initialize a dictionary to store training history
        history = {'total_train_loss': [], 'train_accuracy': []}

        self.logger.info("\nModel loaded and initial Training params setup completed....")
        time.sleep(0.1)

        start_time = time.monotonic()

        # Add metrics header
        self.metrics.append(
            ["num_epochs", "total_time (Sec)", "dataset_size (MBytes)", "read_bw (MB/Sec)", "dataset_read_time (Sec)"])
        tot_read_time, dataload_time, epoch_bw = [], [], []

        # loop over epochs
        self.logger.info("\nTraining the network....")
        for e in tqdm(range(self.epochs)):

            # set the model in training mode
            self.model.train()

            # initialize the total training and validation loss
            totalTrainLoss = 0

            # initialize the number of correct predictions in the training step
            trainCorrect = 0

            self.train_dataloader.dataset.dataset_size_in_bytes.value = 0

            epoch_start_time = time.monotonic()
            epoch_read_time = 0
            dload_elapsed = 0
            # loop over the training set
            for (images, labels, bboxes, img_read_times) in tqdm(self.train_dataloader):
                # send the input to the device
                (images, labels, bboxes) = (images.to(self.device),
                                            labels.to(self.device), bboxes.to(self.device))

                epoch_read_time += torch.mean(img_read_times).item()

                # perform a forward pass and calculate the training loss
                predictions = self.model(images)
                bboxLoss = bboxLossFunc(predictions[0], bboxes)
                classLoss = classLossFunc(predictions[1], labels)
                totalLoss = ((self.config["training"][self.config["training"]["choice"]]["bbox_loss"] * bboxLoss)
                             + (self.config["training"][self.config["training"]["choice"]]["label_loss"] * classLoss))

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                totalLoss.backward()
                opt.step()

                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += totalLoss
                trainCorrect += (predictions[1].argmax(1) == labels).type(torch.float).sum().item()
                dload_elapsed += (time.monotonic() - epoch_start_time)
                epoch_start_time = time.monotonic()

            # calculate the average training loss
            avgTrainLoss = totalTrainLoss / steps_per_epoch

            # calculate the training accuracy
            trainCorrect = trainCorrect / len(self.train_dataloader.dataset)

            # update our training history
            history["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            history["train_accuracy"].append(trainCorrect)

            # print the model training and validation information
            self.logger.info("EPOCH: {}/{}".format(e + 1, self.epochs))
            self.logger.info("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))

            tot_read_time.append(round(epoch_read_time, 4))
            dataload_time.append(round(dload_elapsed, 4))
            dataset_size_mb = round((self.train_dataloader.dataset.dataset_size_in_bytes.value / 1024), 2)
            epoch_bw.append(round((dataset_size_mb / epoch_read_time), 3))
        self.metrics.append([str(self.epochs).replace(',', ';'), str(dataload_time).replace(',', ';'),
                             str(dataset_size_mb).replace(',', ';'),
                             str(epoch_bw).replace(',', ';'), str(tot_read_time).replace(',', ';')])
        total_time = round(time.monotonic() - start_time, 4)
        bw = [round((self.train_dataloader.dataset.dataset_size_in_bytes.value / 1024) / t, 3) for t in tot_read_time]
        train_summary = "** Train Summary **\n"
        train_summary += "\t Epochs:{}, BatchSize:{}, MaxBatchSize:{}\n".format(self.epochs, self.batch_size,
                                                                                self.max_batch_size)
        train_summary += "\t Loading+Training Time:{:.2f} Sec, Dataset Size:{} KBytes, Read_BW:{} MiB/Sec, " \
                         "Total Read Time:{} Secs\n".format(total_time,
                                                            self.train_dataloader.dataset.dataset_size_in_bytes.value,
                                                            bw, tot_read_time)
        self.logger.info(train_summary)

        self.logger.info("\nTraining completed...")

        base_op_dir = (
            self.config["storage"][self.config["storage"]["format"]][self.config["storage"][self.config["storage"]["format"]]["choice"]]["base_output_dir"])
        prediction_path = (
            self.config["storage"][self.config["storage"]["format"]][self.config["storage"][self.config["storage"]["format"]]["choice"]]["predictions_path"])
        plot_path = (
            self.config["storage"][self.config["storage"]["format"]][self.config["storage"][self.config["storage"]["format"]]["choice"]]["plots_path"])
        saved_dir_list = [prediction_path, plot_path]
        saved_model_name = (
            self.config["storage"][self.config["storage"]["format"]][self.config["storage"][self.config["storage"]["format"]]["choice"]]["saved_model_name"])

        if not os.path.exists(base_op_dir):
            for dirc in saved_dir_list:
                os.makedirs(dirc)
        else:
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

        # serialize the model to disk
        saved_model_path = os.path.sep.join([base_op_dir, saved_model_name])
        self.logger.info("Saving the object detector model...")
        torch.save(self.model, saved_model_path)

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history["total_train_loss"], label="Total_Train_Loss")
        plt.plot(history["train_accuracy"], label="Train_Accuracy")
        plt.title("Total Training Loss and Classification Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")

        # save the training plot
        plotPath = os.path.sep.join([plot_path, 'training_plot.png'])
        plt.savefig(plotPath)
        self.logger.info("Training plot saved and script run completed!...")


class PythonReadTrain(DNNTrain):

    def __init__(self, **kwargs):
        self.train_dataloader = kwargs["dataloader"]
        super(PythonReadTrain, self).__init__(**kwargs)

    def train(self):
        # Add metrics header
        self.metrics.append(["time", "dataset_size", "bw"])
        start = time.monotonic()
        for epoch in range(self.epochs):
            for batch_index, data in enumerate(self.train_dataloader):
                if batch_index == self.max_batch_size - 1:
                    break
        dataload_time = round((time.monotonic() - start), 2)
        dataset_size_mb = self.train_dataloader.dataset.dataset_size_in_bytes.value
        bw = round((dataset_size_mb / dataload_time), 2)
        self.metrics.append([str(dataload_time), str(dataset_size_mb), str(bw)])
        train_summary = "** Train Summary **\n"
        train_summary += "\t Epochs:{}, BatchSize:{}, MaxBatchSize:{}\n".format(self.epochs, self.batch_size,
                                                                                self.max_batch_size)
        train_summary += "\t Time:{:.2f} Sec, Detaset Size:{} KBytes, BW:{:.2f} MiB/Sec".format(dataload_time,
                                                                                                self.train_dataloader.dataset.dataset_size_in_bytes.value,
                                                                                                bw)
        self.logger.info(train_summary)


class SequentialDatasetTrain(DNNTrain):
    """
    Example training for sequential dataset.
    #TODO
    """
    def __init__(self, config):
        super(SequentialDatasetTrain, self).__init__(config)

    def train(self):
        pass


class DDPTrain(DNNTrain):
    def __init__(self, config):
        super(DDPTrain, self).__init__(config)

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
        self.name = self.config["training"]["choice"]
        self.logger = kwargs["logger"]
        self.metrics = kwargs["metrics"]
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
                                           metrics=self.metrics,
                                           logger=self.logger
                                           )
            # custom_train.create_model()
            custom_train.train()  # Initiate training
        except Exception as e:
            self.logger.error("{}".format(e))
