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

import argparse
import json
import logging
import os
import time
import utils
from logging.handlers import RotatingFileHandler
from pathlib import Path
from prometheus_client import start_http_server, REGISTRY

import minio_ustat_collector
import minio_rest_collector
import nvmftarget_ustat_collector


APP_NAME = "DSS Metrics Agent"

LOGGING_LEVEL = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARN,
    "ERROR": logging.ERROR,
    "EXCEPTION": logging.ERROR,
    "FATAL": logging.FATAL
}

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

    # get other required metadata
    filter = configs['filter']
    whitelist_patterns = utils.get_whitelist_keys()
    polling_interval_secs = configs['polling_interval_secs']
    num_iterations = 1
    metrics_scopes = {}
    # load metrics scope settings
    with open("metrics_scope.json", "rb") as scopes:
        metrics_scopes = json.loads(scopes.read().decode('UTF-8', "ignore"))

    # need to deserialize escape characters from json
    metrics_scopes = {string.encode().decode('unicode_escape'): scope
                      for string, scope in metrics_scopes.items()}

    # create logger
    logfile_path = configs["logging_file"]
    if not logfile_path:
        raise ValueError("Missing logging_file parameter in config file")

    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    logfile = Path(logfile_path)
    logfile.touch(exist_ok=True)

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(LOGGING_LEVEL[configs['logging_level']])

    log_format = (
        logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s')
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # max size of log file set to 100MB
    file_handler = RotatingFileHandler(logfile_path, mode='a',
                                       maxBytes=100*1024*1024, backupCount=1,
                                       encoding=None, delay=0)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info("Successfully created logger!")

    # expose metrics on promotheus endpoint
    logger.info("\n\n starting server.... \n\n")
    try:
        start_http_server(8000)

        REGISTRY.register(
            minio_rest_collector.MinioRESTCollector(
                configs, metrics_scopes, whitelist_patterns, filter)
        )
        REGISTRY.register(
            nvmftarget_ustat_collector.NVMFTargetUSTATCollector(
                configs, metrics_scopes, polling_interval_secs, num_iterations,
                whitelist_patterns, filter
            )
        )
        REGISTRY.register(
            minio_ustat_collector.MinioUSTATCollector(
                configs, metrics_scopes, polling_interval_secs, num_iterations,
                whitelist_patterns, filter
            )
        )

        while True:
            time.sleep(1)
    except Exception as error:
        logger.info(f"Failed to start Metrics http server: {str(error)}")
