#!/usr/bin/python

import numpy as np
import os,sys
from utils.config import Config, ArgumentParser
import tensorflow as tf
import pathlib

from dataset import DataSet
from datetime import datetime
from framework import TensorFlow, PyTorch

#import tensorflow_datasets as tfds



#tf.enable_eager_execution() # Operation are executed as they are.

class Benchmarking(object):
    def __init__(self, config={}):
        self.config = config
        self.computation = config.get("computation","CPU")
        self.framework = config.get("framework", {})
        self.framework_name = self.framework.get("name", None)
        # dnn framework instance
        self.dnn_framework = None

    def start(self):
        """
        Start the DNN process with dataset creation, training and sample inference.
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
        self.dnn_framework.training()
        self.dnn_framework.inference()

    def stop(self):
        pass


    def metrics(self):
        pass

if __name__ == "__main__":


    print("INFO: Running TensorFlow v{}".format(tf.__version__))
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




