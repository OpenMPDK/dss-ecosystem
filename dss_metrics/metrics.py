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
import multiprocessing
import time

import nvmftarget_collector
import minio_collector
import utils


MetricInfo = namedtuple("MetricInfo", "key, name, value, tags, timestamp")


class MetricsCollector(object):
    def __init__(self, configs):
        self.configs = configs
        self.filter = self.configs['filter']
        self.whitelist_patterns = utils.get_whitelist_keys()

    def collect(self):
        metrics = self.get_metrics()
        for metric in metrics:
            yield metric

    def get_metrics(self):
        manager = multiprocessing.Manager()
        metrics_data_buffer = manager.list()
        metrics = []

        print("collecting metrics from DSS cluster..")

        num_seconds = self.configs['polling_interval_secs']
        num_iterations = 1
        target_obj = nvmftarget_collector.NVMFTargetCollector(
            self.configs['ustat_binary_path'],
            num_seconds,
            num_iterations,
            self.whitelist_patterns,
            filter=self.filter
        )
        minio_obj = minio_collector.MinioCollector(
            self.configs['ustat_binary_path'],
            num_seconds,
            num_iterations,
            self.whitelist_patterns,
            filter=self.filter
        )

        target_proc = multiprocessing.Process(
            target=target_obj.poll_statistics,
            args=[metrics_data_buffer]
        )
        minio_proc = multiprocessing.Process(
            target=minio_obj.poll_statistics,
            args=[metrics_data_buffer]
        )

        target_proc.start()
        minio_proc.start()

        target_proc.join()
        minio_proc.join()

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

    # expose metrics on promotheus endpoint
    print("\n\n starting http server.... \n\n")
    start_http_server(8000)
    REGISTRY.register(
        MetricsCollector(configs)
    )
    while True:
        time.sleep(1)

    # TODO: add logic to insert into prometheus DB
