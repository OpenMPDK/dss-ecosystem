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

import logging
import re
import requests
import socket
import time
import utils

import metrics

from prometheus_client import Metric
from prometheus_client.registry import Collector


class MinioRESTCollector(Collector):
    def __init__(self, configs, metrics_scopes, whitelist_patterns,
                 filter=False):
        self.configs = configs
        self.metrics_scopes = metrics_scopes
        self.whitelist_patterns = whitelist_patterns
        self.filter = filter

        self.ignored_minio_metrics = {
            'minio_http_requests_duration_seconds_bucket',
            'minio_http_requests_duration_seconds_sum',
            'minio_http_requests_duration_seconds_count'}

        self.url_prefix = "http://"
        self.cluster_id_url_suffix = "/minio/cluster_id"
        self.metrics_url_suffix = "/minio/prometheus/metrics"
        self.mc = configs['mc_binary_path']
        self.conf_json_bucket_suffix = configs['conf_json_bucket_suffix']
        self.cluster_id = self.configs['cluster_id']
        self.TYPE = 'minio_rest'
        self.logger = logging.getLogger(metrics.APP_NAME)

    def collect(self):
        self.logger.debug("--- MINIO REST COLLECTOR ---")

        cluster_endpoint_map = utils.get_miniocluster_endpoint_map(self.mc,
                                                                   self.logger)
        cluster_endpoint_map_items = cluster_endpoint_map.items()

        if (not cluster_endpoint_map_items or
           len(cluster_endpoint_map_items) == 0):
            self.logger.error("No MINIO endpoints found")
            raise ValueError("No MINIO endpoints found")

        all_minio_endpoints = []
        for _, endpts in cluster_endpoint_map_items:
            all_minio_endpoints.extend(list(endpts))

        try:
            for minio_endpoint in all_minio_endpoints:
                miniocluster_id = utils.get_minio_cluster_uuid(
                    minio_endpoint, self.url_prefix,
                    self.cluster_id_url_suffix)
                minio_metrics = self.get_minio_metrics_from_endpoint(
                    minio_endpoint)

                if not miniocluster_id or not minio_metrics:
                    self.logger.error("Failed to retrieve MINIO metadata")
                    raise ValueError("Failed to retrieve MINIO metadata")

                tags = {}
                tags['cluster_id'] = self.cluster_id
                tags['target_id'] = socket.gethostname()
                tags['minio_id'] = miniocluster_id
                tags['minio_endpoint'] = minio_endpoint
                tags['type'] = self.TYPE

                for key, value in minio_metrics:
                    if (self.filter and not
                            self.check_whitelist_key(key)):
                        continue

                    # check if scope applies to metric
                    scope = self.get_metric_scope(key)
                    if scope:
                        tags['scope'] = scope

                    name = key
                    metric = Metric(name, key, 'gauge')
                    metric.add_sample(
                        name,
                        value=value,
                        labels=tags,
                        timestamp=time.time()
                    )
                    self.logger.debug(f"collected: {metric}")
                    yield metric

        except Exception as error:
            self.logger.error(
                f"Error: {str(error)} during MINIO REST collection")

    def get_metric_scope(self, key):
        for regex in self.metrics_scopes.keys():
            if re.match(regex, key):
                return self.metrics_scopes[regex]
        return None

    def check_whitelist_key(self, key):
        for regex in self.whitelist_patterns:
            if re.match(regex, key):
                return True
        return False

    def get_minio_metrics_from_endpoint(self, endpoint):
        url = self.url_prefix + endpoint + self.metrics_url_suffix
        r = requests.get(url)
        metrics_data = []
        for line in r.text.splitlines():
            if (line.startswith("minio")
                and not (any(line.startswith(x)
                             for x in self.ignored_minio_metrics))):
                key, val = line.split(" ")
                metrics_data.append((key, float(val)))
        return metrics_data
