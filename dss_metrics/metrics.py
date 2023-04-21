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

from prometheus_client import start_http_server, Summary, REGISTRY
import argparse
import multiprocessing
import os
import time

import nvmf_target
import minio

REQUEST_TIME = Summary(
    'request_processing_seconds',
    'Time spent processing request'
)


def launch_collector(obj, stats_dict):
    root = ''
    if type(obj) is nvmf_target.NVMFTarget:
        root = 'target'
    elif type(obj) is minio.Minio:
        root = 'minio'
    stats_dict[root] = obj.poll_statistics()


def get_blacklist_keys():
    blacklist = []
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = curr_dir + os.sep + 'blacklist.txt'
    with open(file_path) as f:
        blacklist = f.read().splitlines()
    return blacklist


def get_whitelist_keys():
    whitelist = []
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = curr_dir + os.sep + 'whitelist.txt'
    with open(file_path) as f:
        whitelist = f.read().splitlines()
    return whitelist


def get_metrics():
    manager = multiprocessing.Manager()
    stats_dict = manager.dict()

    num_seconds = 1
    num_iterations = 1

    # TODO: read in CLI params / config file
    print("collecting metrics from DSS cluster..")

    target_obj = nvmf_target.NVMFTarget(num_seconds, num_iterations)
    minio_obj = minio.Minio(num_seconds, num_iterations)

    target_proc = multiprocessing.Process(
        target=launch_collector,
        args=[target_obj, stats_dict]
    )
    minio_proc = multiprocessing.Process(
        target=launch_collector,
        args=[minio_obj, stats_dict]
    )

    target_proc.start()
    minio_proc.start()

    target_proc.join()
    minio_proc.join()

    dss_metrics = {}
    for k in dict(stats_dict).copy().keys():
        dss_metrics.update(dict(stats_dict[k]))

    # TODO: populate Metric objects here

    return dss_metrics


# TODO: implement whitelist with REGEX
class MetricsCollector(object):
    def __init__(self, filter=False, whitelist_keys=[]):
        self.filter = filter
        self.whitelist_keys = whitelist_keys

    @REQUEST_TIME.time()
    def collect(self):
        # yield metric
        blacklist_keys = get_blacklist_keys()
        stats = get_metrics()

        # we always filter out black list keys
        for k in stats.copy().keys():
            if k in blacklist_keys:
                stats.pop(k, None)

        # check if whitelist filtering is needed
        if self.filter and self.whitelist_keys:
            for k in stats.copy().keys():
                if k not in whitelist_keys:
                    stats.pop(k, None)

        for metric in stats.values():
            yield metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSS Metrics Agent CLI')
    parser.add_argument(
        "--filter",
        "-fr",
        required=False,
        action='store_true',
        help='Filter DSS metrics to only export metric listed in whitelist.txt'
    )
    args = vars(parser.parse_args())

    whitelist_keys = get_whitelist_keys()

    print("\n\n starting http server.... \n\n")
    start_http_server(8000)
    REGISTRY.register(
        MetricsCollector(filter=args['filter'], whitelist_keys=whitelist_keys)
    )
    while True:
        time.sleep(1)
