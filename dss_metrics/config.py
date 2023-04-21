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

import os
import sys

# TODO: refactor to only leverage what we are using
"""
nvmf_target process name
minio process name
ustat_binary path
logging configs (log level, filename)
promotheus port number
whitelist keys
polling interval

anything else that is used.. 

"""

USTAT_BINARY = "/usr/dss/nkv-sdk/bin/ustat"

CONFIG_SECTIONS = ['agent', 'logging']
PRODUCT_NAME = "nkv-agent"
DFLY_PRODUCT_NAME = "dragonfly"
AGENT_NAME = "agent"
AGENT_CONF_NAME = "%s.conf" % AGENT_NAME
AGENT_LOG_NAME = "%s.log" % AGENT_NAME

USTAT_NAME = "ustat"

BINARY_DIR = "/usr/%s/" % DFLY_PRODUCT_NAME
LOG_DIR = "/var/log/%s/" % PRODUCT_NAME
CONFIG_DIR = "/etc/%s/" % PRODUCT_NAME
TARGET_CONF_DIR = "/etc/%s/" % DFLY_PRODUCT_NAME

TARGET_BIN_NAME = "nvmf_tgt"
TARGET_CONF_NAME = "nvmf.in.conf"
TARGET_LOG_NAME = "nvmf_err.log"


DEFAULT_SETTINGS = {
    CONFIG_SECTIONS[0]: {
        "nvmf_tgt": BINARY_DIR + TARGET_BIN_NAME,
        "nvmf_conf_file": TARGET_CONF_DIR + TARGET_CONF_NAME,
        "ustat_binary": BINARY_DIR + USTAT_NAME,
        "hugepages": "8192",
        "stats_proto": "graphite",
        "stats_server": "127.0.0.1",
        "stats_poll": "10",
        "stats_port": "2004",
    },
    CONFIG_SECTIONS[1]: {
        "log_dir": LOG_DIR,
        "log_file": AGENT_LOG_NAME,
        "log_level": "DEBUG",
        "console": "enabled",
        "console_level": "INFO",
        "syslog": "enabled",
        "syslog_level": "DEBUG",
        "syslog_facility": "local0",
    },
}


def create_config(filename):
    """
    Create daemon's configuration file if
    it does not exist
    :filename:
    :return:
    """
    settings = DEFAULT_SETTINGS

    if os.path.exists(filename):
        return None

    default_conf = ""
    for section in CONFIG_SECTIONS:
        if section in settings:
            for idx, key in enumerate(sorted(settings[section])):
                if idx == 0:
                    default_conf += "[%s]\n" % section
                default_conf += "%s=%s\n" % (key, settings[section][key])
            default_conf += "\n"

    print("Creating default CLI configuration file '%s'" % filename)
    try:
        f = open(filename, 'wb')
    except Exception as e:
        print("Open %s failed - %s" % (filename, str(e)))
        sys.exit(-1)
    f.write(default_conf)
    f.close()

    return settings
