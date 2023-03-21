#!/usr/bin/python

"""
# The Clear BSD License
#
# Copyright (c) 2022 Samsung Electronics Co., Ltd.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Samsung Electronics Co., Ltd. nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import json
import argparse


class Config(object):

    def __init__(self, params, config_filepath=None):
        self.config_file = config_filepath or self.get_config_file(params.get("config"))
        self.params = params
        self.config = self.process_config()

    def get_config(self):
        """
        Get configuration details from config file ...
        :return:<dict> complete configuration dictionary.
        """
        return self.config

    def process_config(self):
        config = {}
        with open(self.config_file, "rb") as cfg:
            config = json.loads(cfg.read().decode('UTF-8', "ignore"))
        if self.params:
            for param in self.params:
                if self.params[param] is not None:  # must explicitly check for None since we still want False values to be overriden onto the config dict
                    config[param] = self.params[param]
        # process compaction options accordingly
        config['compaction'] = True if 'compaction' not in config else config['compaction'] == 'yes'
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


def commandLineArgumentParser():
    parser = argparse.ArgumentParser(description='Distributed version of S3 CLI to perform PUT,LIST,GET,DEL operations')
    subparser = parser.add_subparsers(help="Supported Operations ... ")

    put_parser = subparser.add_parser("PUT", help="Upload the files to S3 storage")
    list_parser = subparser.add_parser("LIST", help="List the buckets/objects from S3 storage!")
    get_parser = subparser.add_parser("GET", help="Download the files from S3 storage bucket!")
    del_parser = subparser.add_parser("DEL", help="Remove the objects from the S3 storage bucket!")

    # All arguments for PUT
    put_parser.add_argument("--thread", "-t", type=int, default=1, required=False,
                            help='Specify number of Jobs to be used for parallel processing. ')
    put_parser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
    put_parser.add_argument("--cluster", "-c", type=str, nargs="+", default="10.1.51.2", required=True,
                            help='Specify cluster name  ...')
    put_parser.add_argument("--prefix", "-p", type=str, required=False,
                            help='Specify operation type such as read=r write=w , wr...')
    put_parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')

    # All arguments for LIST
    list_parser.add_argument("--thread", "-t", type=int, default=1, required=False,
                             help='Specify number of Jobs to be used for parallel processing. ')
    list_parser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
    list_parser.add_argument("--cluster", "-c", type=str, nargs="+", default="10.1.51.2", required=True,
                             help='Specify cluster name  ...')
    list_parser.add_argument("--prefix", "-p", type=str, required=False,
                             help='Specify operation type such as read=r write=w , wr...')
    list_parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')

    # All arguments for GET
    get_parser.add_argument("--thread", "-t", type=int, default=1, required=False,
                            help='Specify number of Jobs to be used for parallel processing. ')
    get_parser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
    get_parser.add_argument("--cluster", "-c", type=str, nargs="+", default="10.1.51.2", required=True,
                            help='Specify cluster name  ...')
    get_parser.add_argument("--prefix", "-p", type=str, required=False,
                            help='Specify operation type such as read=r write=w , wr...')
    get_parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')

    # All arguments for DEL
    del_parser.add_argument("--thread", "-t", type=int, default=1, required=False,
                            help='Specify number of Jobs to be used for parallel processing. ')
    del_parser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
    del_parser.add_argument("--cluster", "-c", type=str, nargs="+", default="10.1.51.2", required=True,
                            help='Specify cluster name  ...')
    del_parser.add_argument("--prefix", "-p", type=str, required=False,
                            help='Specify operation type such as read=r write=w , wr...')
    del_parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')

    options = parser.parse_args()

    return sys.argv[1:2][0], vars(options)


def ClientApplicationArgumentParser():
    parser = argparse.ArgumentParser(description='Distributed version of S3 CLI to perform PUT,LIST,GET,DEL operations')
    parser.add_argument("--client_id", "-id", type=int, default=1, required=True,
                        help='Specify client node IP address ')
    parser.add_argument("--operation", "-op", type=str, required=True,
                        help='Specify operation such as PUT,LIST,DEL,GET')
    parser.add_argument("--ip_address", "-ip", type=str, required=True,
                        help='Specify Client Node IP address')
    parser.add_argument("--master_node", "-mn", required=False, action='store_true',
                        help='Is client running on same node of master?')
    parser.add_argument("--config", "-cfg", type=str, required=False, help='Specify configuration file path')
    parser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                        help='Dry run - Just check operation is working , but does not actual upload')
    parser.add_argument("--port_index", "-pi", type=str, required=True, help='Specify index port')
    parser.add_argument("--port_status", "-ps", type=str, required=True, help='Specify status port')
    parser.add_argument("--dest_path", "-dp", type=str, required=False,
                        help='Specify Destination Directory for GET operation only')
    parser.add_argument("--debug", "-d", required=False, action='store_true',
                        help='Run DataMover in debug mode')
    parser.add_argument("--skip_upload", "-su", required=False, action='store_true',
                        help='Skip data upload operation for DataIntegrity')
    parser.add_argument("--distributed", "-dist", required=False, action='store_true',
                        help='Enable distributed LISTing')
    parser.add_argument("--stop", "-stop", required=False, action='store_true',
                        help='Stop Client application processes gracefully')

    options = vars(parser.parse_args())
    return options


class CommandLineArgument(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Distributed version of S3 CLI to perform PUT,LIST,GET,DEL operations!')
        subparsers = parser.add_subparsers(help="Supported Operations ... ")
        put_parser = subparsers.add_parser("PUT", help="Upload the files to S3 storage")
        list_parser = subparsers.add_parser("LIST", help="List the objects from S3 storage!")
        get_parser = subparsers.add_parser("GET", help="Download the files from S3 storage!")
        del_parser = subparsers.add_parser("DEL", help="Remove the objects from the S3 storage!")
        test_parser = subparsers.add_parser("TEST", help="Perform DataMover data_integrity test.")

        if not sys.argv[1:2]:
            parser.print_help()
            sys.exit()

        self.operation = sys.argv[1:2][0]

        if self.operation.upper() == "PUT":
            self.put(put_parser)
        elif self.operation.upper() == "GET":
            self.get(get_parser)
        elif self.operation.upper() == "LIST":
            self.list(list_parser)
        elif self.operation.upper() == "DEL":
            self.delete(del_parser)
        elif self.operation.upper() == "TEST":
            self.test(test_parser)
        else:
            parser.print_help()
            sys.exit()

        self.options = vars(parser.parse_args())
        # convert user input to lower case
        self.options = {k: v.lower() if type(v) is str else v for k, v in self.options.items()}

    def put(self, subparser):
        subparser.add_argument("--thread", "-t", type=int, default=1, required=False,
                               help='Specify number of Jobs to be used for parallel processing. ')
        subparser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
        subparser.add_argument("--prefix", "-p", type=str, required=False,
                               help='Specify object-key prefix, should be <nfs server ip>/<any prefix key>/')
        subparser.add_argument("--config", "-cfg", type=str, default='/etc/dss/datamover/config.json', required=False, help='Specify configuration file path')
        subparser.add_argument("--compaction", "-com", type=str, default=argparse.SUPPRESS, required=False, help='Enable target compaction')
        subparser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                               help='Dry run - Just check operation is working , but does not actual upload')
        subparser.add_argument("--debug", "-d", required=False, action='store_true',
                               help='Run DataMover in debug mode')
        subparser.add_argument("--profile", "-pro", required=False, action='store_true',
                               help='Profiling of PUT operation (Not Implemented)')
        subparser.add_argument("--standalone", "-sa", required=False, action='store_true',
                               help='Run in standalone mode')

    def get(self, subparser):
        subparser.add_argument("--thread", "-t", type=int, default=1, required=False,
                               help='Specify number of Jobs to be used for parallel processing. ')
        subparser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
        subparser.add_argument("--prefix", "-p", type=str, required=False,
                               help='Specify object-key prefix, should be <nfs server ip>/<any prefix key>/')
        subparser.add_argument("--config", "-cfg", type=str, default='/etc/dss/datamover/config.json', required=False, help='Specify configuration file path')
        subparser.add_argument("--dest_path", "-dp", type=str, required=True, help='Specify destination file path')
        subparser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                               help='Dry run - Just check operation is working , but does not actual download')
        subparser.add_argument("--debug", "-d", required=False, action='store_true',
                               help='Run DataMover in debug mode')
        subparser.add_argument("--profile", "-pro", required=False, action='store_true',
                               help='Profiling of GET operation (Not Implemented)')

    def list(self, subparser):
        subparser.add_argument("--thread", "-t", type=int, default=1, required=False,
                               help='Specify number of Jobs to be used for parallel processing. ')
        subparser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
        subparser.add_argument("--prefix", "-p", type=str, required=False,
                               help='Specify object-key prefix, should be <nfs server ip>/<any prefix key>/')
        subparser.add_argument("--config", "-cfg", type=str, default='/etc/dss/datamover/config.json', required=False, help='Specify configuration file path')
        subparser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                               help='Dry run - Just check operation is working , but does not actual listing')
        subparser.add_argument("--debug", "-d", required=False, action='store_true',
                               help='Run DataMover in debug mode')
        subparser.add_argument("--dest_path", "-dp", type=str, required=False,
                               help='Path to store object keys in a file.')
        subparser.add_argument("--distributed", "-dist", required=False, action='store_true',
                               help='Perform LISTing in distributed mode')
        subparser.add_argument("--profile", "-pro", required=False, action='store_true',
                               help='Profiling of LIST operation (Not Implemented)')

    def delete(self, subparser):
        subparser.add_argument("--thread", "-t", type=int, default=1, required=False,
                               help='Specify number of Jobs to be used for parallel processing. ')
        subparser.add_argument("--bucket", "-b", type=str, required=False, help='Specify bucket name.. ')
        subparser.add_argument("--prefix", "-p", type=str, required=False,
                               help='Specify object-key prefix, should be <nfs server ip>/<any prefix key>/')
        subparser.add_argument("--compaction", "-com", type=str, default=argparse.SUPPRESS, required=False, help='Enable target compaction')
        subparser.add_argument("--config", "-cfg", type=str, default='/etc/dss/datamover/config.json', required=False, help='Specify configuration file path')
        subparser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                               help='Dry run - Just check operation is working , but does not actual delete')
        subparser.add_argument("--debug", "-d", required=False, action='store_true',
                               help='Run DataMover in debug mode')
        subparser.add_argument("--profile", "-pro", required=False, action='store_true',
                               help='Profiling of DEL operation (Not Implemented)')

    def test(self, subparser):
        subparser.add_argument("--data_integrity", "-di", required=True, action='store_true',
                               help='Run DataMover data integrity test')
        subparser.add_argument("--skip_upload", "-su", required=False, action='store_true',
                               help='Skip data upload operation')
        subparser.add_argument("--prefix", "-p", type=str, required=False,
                               help='Specify object-key prefix, should be <nfs server ip>/<any prefix key>/')
        subparser.add_argument("--config", "-cfg", type=str, default='/etc/dss/datamover/config.json', required=False, help='Specify configuration file path')
        subparser.add_argument("--debug", "-d", required=False, action='store_true',
                               help='Run DataMover in debug mode')
        subparser.add_argument("--dest_path", "-dp", type=str, required=True, help='Specify destination file path')

    def get_operation(self):
        return sys.argv[1:2][0]


def TargetCompactionArgumentParser():
    parser = argparse.ArgumentParser(description='DSS Target compaction!')
    parser.add_argument("--ip_address", "-ip", type=str, required=True,
                        help='Specify Target Node IP address')
    parser.add_argument("--user_id", "-u", type=str, required=True,
                        help='Userid')
    parser.add_argument("--password", "-p", type=str, required=False,
                        help='Password for userid')
    parser.add_argument("--subsystem_nqn", "-nqn", type=str, required=False, help='Specify subsystem-nqn in comma separated form')
    parser.add_argument("--logdir", "-log", type=str, default="/var/log/dss", required=False,
                        help='A path where compaction log gets created')
    parser.add_argument("--installation_path", "-i", type=str, default="/usr/dss/nkv-target", required=False,
                        help='Target software installation path')
    parser.add_argument("--dryrun", "-dr", required=False, action='store_true',
                        help='Dry run - Just check operation is working , but does not actual upload')
    parser.add_argument("--debug", "-d", required=False, action='store_true',
                        help='Run DataMover compaction in debug mode')

    options = vars(parser.parse_args())
    return options
