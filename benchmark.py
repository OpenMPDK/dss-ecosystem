#!/usr/bin/python

import numpy as np
import os,sys
from utils.config import Config, ArgumentParser

import pathlib

from datetime import datetime
from framework import TensorFlow, PyTorch


class Benchmarking(object):
    def __init__(self, config={}):
        self.config = config
        self.computation = config.get("computation","CPU")
        self.framework = config.get("framework", {})
        self.execution_config = config["execution"]
        self.framework_name = self.framework.get("name", None)
        # dnn framework instance
        self.dnn_framework = None

    def start(self):
        """
        Start the DNN process with dataset creation, training and sample inference.
        The above three steps can be controlled with switches.
        "Initialization": Create dataset always. (default)
        "Training": Optional. In future, we can load saved trained model
        "Inference": Optional
        :return:
        """
        # Create a FrameWork
        if self.framework_name == "TensorFlow":
            self.dnn_framework =  TensorFlow(self.config)
        elif self.framework_name == "PyTorch":
            self.dnn_framework = PyTorch(self.config)
        else:
            pass

        print(f"INFO: Starting DNN benchmark with {self.framework_name} framework")
        self.dnn_framework.initialize()
        if self.execution_config["steps"]["model"]:
            self.dnn_framework.create_model()
        if self.execution_config["steps"]["training"]:
            self.dnn_framework.training()

        if self.execution_config["steps"]["inference"]:
            self.dnn_framework.inference()

    def stop(self):
        pass


    def metrics(self):
        pass

if __name__ == "__main__":



    #train_image()
    #sys.exit()
    # Process arguments
    params = ArgumentParser()
    print(params)
    config_obj = Config(params)
    config = config_obj.get_config()
    print(config)

    bench = Benchmarking(config)
    bench.start()
    #print(len(bench.dataset))
    #print(bench.dataset[0:5])




