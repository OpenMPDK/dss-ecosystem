#!/usr/bin/python
"""
 *   BSD LICENSE
 *
 *   Copyright (c) 2021 Samsung Electronics Co., Ltd.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Samsung Electronics Co., Ltd. nor the names of
 *       its contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import json
import argparse


class Config(object):

    def __init__(self, params):
        self.config_file = self.get_config_file(params.get("config"))
        self.params = params
        self.config = self.process_config()

    def get_config(self):
        """
        Get configuration details from config file ...
        :return:<dict> complete configuration dictionary.
        """
        return self.config

    def process_config(self):
        """
        Update configuration based on
        :return:
        """
        config = {}
        with open(self.config_file, "rb") as cfg:
            config = json.loads(cfg.read().decode('UTF-8', "ignore"))
        # Update config location
        if self.params["config"] is None:
            config["config"] = self.config_file
        if self.params:
            for param in self.params:
                if param in config:
                    if self.params[param]:
                        config[param] = self.params[param]
                else:
                    # dataloader_workers
                    if self.params[param]:
                        if param == "dataloader_workers":
                            config["framework"]["PyTorch"]["DataLoader"]["num_workers"] = self.params[param]
                        elif param == "listing_workers":
                            config["execution"]["workers"] = self.params[param]
                        elif param == "batch_size":
                            config["framework"]["batch_size"] = self.params[param]
                        elif param == "max_batch_size":
                            config["framework"]["max_batch_size"] = self.params[param]
                        else:
                            config[param] = self.params[param]
        return config

    def get_config_file(self, config_file):
        """
        Return the configuration file.
        :param config_file:
        :return:<string> complete configuration file
        """
        if not config_file:
            config_file = os.path.dirname(__file__) + "/../config/config.json"
            config_file = os.path.abspath(config_file)
            print("INFO: Using configuration file from {}".format(config_file))
        return config_file


def ArgumentParser():
    parser = argparse.ArgumentParser(description='Benchmarking tool')
    parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')
    parser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                        help='Dry run - Just check operation is working , but does not actual upload')
    parser.add_argument("--workers", "-w", type=int, required=False,
                        help='Workers ')
    parser.add_argument("--debug", "-d", required=False, action='store_true',
                        help='Run benchmark tool in debug mode')

    options = vars(parser.parse_args())
    return options
