"""
The Clear BSD License

Copyright (c) 2023 Samsung Electronics Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Samsung Electronics Co., Ltd. nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from collections import namedtuple
from prometheus_client import start_http_server, Summary, REGISTRY, Metric
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
import time
import utils

import minio_ustat_collector
import minio_rest_collector
import nvmftarget_ustat_collector

MetricInfo = namedtuple("MetricInfo", "key, name, value, tags, timestamp")
COLLECTOR_TIMEOUT = 120


class MetricsCollector(object):
    def __init__(self, configs):
        self.configs = configs
        self.filter = self.configs['filter']
        self.whitelist_patterns = utils.get_whitelist_keys()
        self.exit_flag = multiprocessing.Event()
        self.logger = logging.getLogger('root')

    def collect(self):
        metrics = self.get_metrics()
        for metric in metrics:
            yield metric

    def create_collector_proc(self, name, obj, metrics_data_buffer, exit_flag):
        try:
            proc = multiprocessing.Process(
                name=name,
                target=obj.poll_statistics,
                args=[metrics_data_buffer, exit_flag]
            )
            return proc
        except Exception as error:
            logger.error(f"Error launching collector {error}")

    def get_metrics(self):
        manager = multiprocessing.Manager()
        metrics_data_buffer = manager.list()
        metrics = []

        logger.info("collecting metrics from DSS cluster..")

        num_seconds = self.configs['polling_interval_secs']
        num_iterations = 1

        collector_procs = []

        # initialize collectors
        target_ustat_obj = nvmftarget_ustat_collector.NVMFTargetUSTATCollector(
            self.configs, num_seconds, num_iterations, self.whitelist_patterns,
            filter=self.filter
        )
        minio_ustat_obj = minio_ustat_collector.MinioUSTATCollector(
            self.configs, num_seconds, num_iterations, self.whitelist_patterns,
            filter=self.filter
        )
        minio_rest_obj = minio_rest_collector.MinioRESTCollector(
            self.configs, self.whitelist_patterns, filter=self.filter
        )

        # initialize collector processes
        collector_procs.append(self.create_collector_proc(
            "target_ustat", target_ustat_obj, metrics_data_buffer,
            self.exit_flag))
        collector_procs.append(self.create_collector_proc(
            "minio_ustat", minio_ustat_obj, metrics_data_buffer,
            self.exit_flag))
        collector_procs.append(self.create_collector_proc(
            "minio_rest", minio_rest_obj, metrics_data_buffer,
            self.exit_flag))

        # start collectors
        for proc in collector_procs:
            try:
                proc.start()
            except Exception as error:
                logger.error(f"Failed to start {proc.name}:{str(error)}")

        # run collectors for specified duration
        time.sleep(self.configs["metrics_agent_runtime_per_interval"])

        # send exit flag to stop collectors
        self.exit_flag.set()

        # wait for collectors to finish and kill any remaining collectors
        for proc in collector_procs:
            try:
                proc.join(COLLECTOR_TIMEOUT)
                if proc.is_alive():
                    logger.warning(f"process {proc.name} hanging, terminating")
                    proc.terminate()
            except Exception as error:
                logger.error(f"Failed to terminate {proc.name}: {str(error)}")

        # populate Prometheus metric objects
        for m in metrics_data_buffer:
            metric = Metric(m.name, m.key, 'gauge')
            metric.add_sample(
                m.name,
                value=m.value,
                labels=m.tags,
                timestamp=m.timestamp
            )
            metrics.append(metric)
        return metrics


if __name__ == '__main__':

    # load CLI args
    parser = argparse.ArgumentParser(description='DSS Metrics Agent CLI')
    parser.add_argument(
        "--filter",
        "-fr",
        required=False,
        action='store_true',
        help='Filter DSS metrics to only export metric listed in whitelist.txt'
    )
    parser.add_argument(
        "--config",
        "-cfg",
        required=True,
        type=str,
        help='Specify configuration file path'
    )
    cli_args = vars(parser.parse_args())

    # load config file args
    configs = {}
    with open(cli_args['config'], "rb") as cfg:
        configs = json.loads(cfg.read().decode('UTF-8', "ignore"))

    # merge configs
    configs.update(cli_args)

    # create logger
    logfile_path = configs["logging_file"]
    if not logfile_path:
        raise ValueError("Missing logging_file parameter in config file")

    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    if not os.path.exists(logfile_path):
        os.mknod(logfile_path)

    log_format = (
        logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s')
    )
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)

    # max size of log file set to 100MB
    file_handler = RotatingFileHandler(logfile_path, mode='a',
                                       maxBytes=100*1024*1024, backupCount=1,
                                       encoding=None, delay=0)
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Successfully created logger!")

    # expose metrics on promotheus endpoint
    logger.info("\n\n starting http server.... \n\n")
    try:
        start_http_server(8000)
        REGISTRY.register(
            MetricsCollector(configs)
        )
        while True:
            time.sleep(1)
    except Exception as error:
        logger.info(f"Failed to start Metrics http server: {str(error)}")
