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

import json
import metrics
import re
import requests
import socket
import subprocess
import time


class MinioRESTCollector(object):
    def __init__(self, configs, whitelist_patterns, filter=False):
        self.configs = configs
        self.whitelist_patterns = whitelist_patterns
        self.filter = filter
        self.minio_metrics = {'minio_disk_storage_used_bytes'}
        self.metrics_url_prefix = "http://"
        self.metrics_url_suffix = "/minio/prometheus/metrics"
        self.mc = configs['mc_binary_path']
        self.conf_json_bucket_suffix = configs['conf_json_bucket_suffix']
        self.cluster_id = self.configs['cluster_id']
        self.TYPE = 'minio'

    def poll_statistics(self, metrics_data_buffer):
        cluster_endpoint_map = self.get_miniocluster_endpoint_map()
        for id, endpts in cluster_endpoint_map.items():
            miniocluster_id = id
            minio_endpoint = endpts.pop()
            minio_metrics = self.get_minio_metrics_from_endpoint(
                minio_endpoint)

            tags = {}
            tags['cluster_id'] = self.cluster_id
            tags['target_id'] = socket.gethostname()
            tags['minio_id'] = miniocluster_id
            tags['type'] = self.TYPE

            for metric in minio_metrics:
                if self.filter and not self.check_whitelist_key(metric[0]):
                    continue
                metrics_data_buffer.append(
                    metrics.MetricInfo(
                        metric[0], metric[0], metric[1], tags, time.time())
                )

    def check_whitelist_key(self, key):
        for regex in self.whitelist_patterns:
            if re.match(regex, key):
                return True
        return False

    def get_miniocluster_endpoint_map(self):
        proc = subprocess.Popen(
            [self.mc, 'config', 'host', 'list'],
            stdout=subprocess.PIPE
        )
        local_minio_host = None
        miniocluster_endpoint_map = dict()  # {<miniocluster> : {<endpoints>} }
        try:
            for line in proc.stdout.readlines():
                # find local minio host or endpoint
                decoded_line = line.decode('utf-8').strip("\n")
                if decoded_line.startswith('local_'):
                    local_minio_host = decoded_line.strip()
                    break
        except Exception as error:
            print(f"Unable to read host list: {str(error)}")
            return {}

        if local_minio_host:
            conf_json_path = local_minio_host + self.conf_json_bucket_suffix
            try:
                proc = subprocess.Popen([self.mc, 'cat', conf_json_path],
                                        stdout=subprocess.PIPE)
                dss_conf_dict = json.loads(
                    proc.communicate()[0].decode('utf-8'))

                for cluster in dss_conf_dict["clusters"]:
                    minio_cluster_id = cluster["id"]
                    minio_endpoints = set()
                    for endpoint_info in cluster["endpoints"]:
                        minio_endpoints.add(
                            endpoint_info["ipv4"] + ":"
                            + str(endpoint_info["port"])
                        )
                    miniocluster_endpoint_map[minio_cluster_id] = (
                        minio_endpoints)
            except KeyError as error:
                print(
                    f"conf.json missing cluster information: {str(error)}"
                )
            except Exception as error:
                print(f"Error when processing conf.json: {str(error)}")
        else:
            raise ValueError("Unable to find local minio host/endpoint")

        return miniocluster_endpoint_map

    def get_minio_metrics_from_endpoint(self, endpoint):
        url = self.metrics_url_prefix + endpoint + self.metrics_url_suffix
        r = requests.get(url)
        metrics_data = []
        for line in r.text.splitlines():
            if (any(line.startswith(metric) for metric in self.minio_metrics)):
                key, val = line.split(" ")
                metrics_data.append((key, float(val)))
        return metrics_data
