#!/usr/bin/python

import os
import sys
from utils.config import Config, ArgumentParser
from utils import __VERSION__

import pathlib

from datetime import datetime
from framework import TensorFlow, PyTorch
from logger import MultiprocessingLogger
from metrics.metrics import Metrics
from multiprocessing import Queue, Value


class Benchmarking(object):
    def __init__(self, config={}):
        self.config = config
        self.computation = config.get("computation", "CPU")
        self.framework = config.get("framework", {})
        self.execution_config = config["execution"]
        self.framework_name = self.framework.get("name", None)
        # dnn framework instance
        self.dnn_framework = None

        # Logging
        self.logging_path = "/var/log/dss"
        self.logging_level = "INFO"
        if "logging" in config:
            self.logging_path = config["logging"].get("path", "/var/log/dss")
            self.logging_level = config["logging"].get("level", "INFO")
        if config.get("debug", False):
            self.logging_level = "DEBUG"
        self.logger = None
        self.logger_status = Value('i', 0)  # 0=NOT-STARTED,1=RUNNING,2=STOPPED
        self.logger_queue = Queue()

        # Metrics
        self.metric = None

    def start(self):
        """
        Start the DNN process with dataset creation, training and sample
        inference. The above three steps can be controlled with switches.
        "Initialization": Create dataset always. (default)
        "Training": Optional. In future, we can load saved trained model
        "Inference": Optional
        :return:
        """
        # Start Logger
        self.start_logging()
        # Create a FrameWork
        if self.framework_name == "TensorFlow":
            self.dnn_framework = TensorFlow(self.config, self.logger)
        elif self.framework_name == "PyTorch":
            self.dnn_framework = PyTorch(config=self.config,
                                         logger=self.logger)
        else:
            pass

        self.logger.info(f"Starting DNN benchmark with {self.framework_name} \
                          framework")
        self.dnn_framework.initialize()

        # Create model
        if self.execution_config["steps"]["model"]:
            self.dnn_framework.create_model()
        # Train model
        if self.execution_config["steps"]["training"]:
            self.dnn_framework.training()
        # Perform inference
        if self.execution_config["steps"]["inference"]:
            self.dnn_framework.inference()

        # Initiate metrics collection
        if self.execution_config["steps"]["metrics"]:
            self.metrics_collection()

    def stop(self):
        pass

    def metrics_collection(self):
        """
        Create metrics collection class
        :return:
        """
        self.dnn_framework.update_metrics()
        self.metric = Metrics(config=self.config["metrics"],
                              data=self.dnn_framework.metrics,
                              logger=self.logger)
        self.metric.process()

    def start_logging(self):
        """
        Start Multiprocessing logger
        :return:
        """
        self.logger = MultiprocessingLogger(self.logger_queue,
                                            self.logger_status
                                            )
        self.logger.config(self.logging_path,
                           __file__,
                           self.logging_level)
        self.logger.start()
        self.logger.info("** DSS_DNN_Bench VERSION:{} **".format(__VERSION__))
        self.logger.info("Started Logger with {} \
                         mode!".format(self.logging_level))

    def stop_logging(self):
        """
        Stop multiprocessing logger
        :return:
        """
        self.logger.stop()


if __name__ == "__main__":

    # train_image()
    # sys.exit()
    # Process arguments
    params = ArgumentParser()
    # print(params)
    config_obj = Config(params)
    config = config_obj.get_config()
    # print(config)

    bench = Benchmarking(config)
    bench.start()
    bench.stop_logging()
