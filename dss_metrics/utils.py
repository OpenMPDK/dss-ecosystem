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
import os
import psutil
import re
import requests
import socket
import subprocess
import time
import ast


def valid_ip(address):
    try:
        socket.inet_aton(address)
        return True
    except Exception as error:
        print(f"Invalid ip: {str(error)}")
        return False


def flat_dict_generator(indict, pre=None):
    """
    Recursive dictionary object to generator. Allows
    flat traversal of all dictionary key-values.
    :param indict: Recursive lower level of dict.
    :param pre: Parent keys to concatenate with child keys.
    :return:
    """
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in flat_dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in flat_dict_generator(v, pre + [key]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        list_value = pre
        list_value.append(indict)
        yield list_value
        # yield indict


def read_linux_file(path):
    buf = None
    try:
        with open(path) as f:
            buf = str((f.read().rstrip()))
    except Exception as error:
        print(f"failed to read linux file: {str(error)}")
    return buf


def time_delta(timestamp):
    now = time.time()
    elapsed = int(now - float(timestamp))
    return elapsed


class KVLog:
    SUCCESS = 1
    WARN = 2
    ERROR = 3

    @staticmethod
    def kvprint(status, msg):
        if status == KVLog.SUCCESS:
            color = "1;37;42"
        elif status == KVLog.WARN:
            color = "1;37;43"
        elif status == KVLog.ERROR:
            color = "1;37;41"
        else:
            color = "0"
        print(('\x1b[%sm' % color) + msg + '\x1b[0m')


def match_cmds_with_process(cmds, proc_args, lazy_regex):
    match = True

    if not cmds:
        return match

    for cmd in cmds:
        if cmd not in proc_args:
            m = None
            if lazy_regex:
                for arg in proc_args:
                    m = re.match(r"^.*?%s.*?$" % cmd, arg)
                    if m is not None:
                        break
            if m is None:
                match = False
                break
        else:
            proc_args.remove(cmd)
    return match


def pidfile_is_running(regex, cmds, pidfile):
    """
    Try to open pid file to read the pid.
    Check if process with read pid is running.
    See if the running process matches our regex and
    cmds we expect it to have.
    :param regex: Process name regex to match.
    :param cmds: Process arguments to compare and match.
    :param pidfile: Path to read/write pid file.
    :return: Running PID otherwise 0.
    """
    if os.path.isfile(pidfile):
        try:
            fh = open(pidfile, 'rb')
        except Exception as e:
            raise e
        try:
            pid_rd = int(fh.read())
        except Exception as e:
            fh.close()
            return 0
        try:
            p = psutil.Process(pid_rd)
        except psutil.NoSuchProcess:
            return 0
        except Exception as e:
            raise e

        p_name_found = cmds_match = False
        m = re.match(regex, p.name())
        if m is not None:
            p_name_found = True
            cmds_match = match_cmds_with_process(cmds, p.cmdline(), 1)

        if p_name_found and cmds_match:
            return pid_rd
    return 0


def find_process_pid(pname_regex, cmds):
    """
    Detect if given process name is running with
    given cmds.
    :param pname_regex: Process name regex to match.
    :param cmds: Process cmd regex to match.
    :return: Running PID otherwise 0.
    """
    # Ignore finding this process
    cur_pid = os.getpid()
    # Ignore finding the parent of this process
    par_pid = os.getppid()
    pid_list = []
    for proc in psutil.process_iter(attrs=['ppid', 'pid', 'name']):
        if cur_pid == proc.info["pid"] or \
           par_pid == proc.info["pid"]:
            continue

        p_name_found = cmds_match = False
        m = re.match(pname_regex, proc.info["name"])
        if m is not None:
            p_name_found = True
            cmds_match = match_cmds_with_process(cmds, proc.cmdline(), 1)

        if p_name_found and cmds_match:
            pid_list.append(proc.info["pid"])

    return pid_list


def check_spdk_running(nvmf_tgt_pidfile='nvmf_tgt.pid'):
    """
    Return if given process is running and its pid.
    :return: process_running (0/1), pid
    """
    process_running = 0
    proc_name = re.compile(r"^reactor_\d+$")
    pid = pidfile_is_running(proc_name, [], nvmf_tgt_pidfile)
    if pid > 0:
        process_running = 1
    else:
        cmds = ["nvmf_tgt", ]
        pid_list = find_process_pid(proc_name, cmds)
        if pid_list:
            pid = pid_list[0]
            if pid > 0:
                process_running = 1
    return pid, process_running


def _eval(string_value=None):
    """
    Convert string value to strings, bytes, numbers, tuples,
    lists, dicts, sets, booleans, and None

    :param string_value:
    :return:
    """
    try:
        string_value = ast.literal_eval(string_value)
    except Exception as e:
        print(str(e))

    return string_value


def convert_stats_dict_to_json_dict(stats_dict):
    json_dict = dict()
    for k, v in stats_dict.items():
        keys = k.split('.')
        json_kv = json_dict
        for key in keys[:-1]:
            json_kv = json_kv.setdefault(key, {})
        json_kv[keys[-1]] = v
    return json_dict


def get_device_subsystem_map():
    device_subsystem_map = {}
    root_dir = '/sys/devices/virtual/nvme-subsystem'
    if not os.access(root_dir, os.X_OK):
        return
    for root, dirs, files in os.walk(root_dir):
        if not root.startswith(os.path.join(root_dir, 'nvme-subsys')):
            continue
        for d in dirs:
            if d.startswith('nvme'):
                subsys_path = os.path.join(root, d)
                with open(subsys_path + '/subsysnqn') as f:
                    nqn = f.readline().strip()
                with open(subsys_path + '/serial') as f:
                    serial = f.readline().strip()
                device_subsystem_map[d + 'n1'] = {'nqn': nqn, 'serial': serial}
    return device_subsystem_map


def get_whitelist_keys():
    whitelist = []
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = curr_dir + os.sep + 'whitelist.txt'
    with open(file_path) as f:
        whitelist = f.read().splitlines()
    return whitelist


def get_minio_endpoint_from_process(pid, logger):
    proc_filepath = f"/proc/{pid}/cmdline"

    proc = subprocess.Popen(
        ['cat', proc_filepath],
        stdout=subprocess.PIPE
    )
    minio_cmd = proc.stdout.read().decode("utf-8")
    endpts_found = re.findall("--address(.*?)(?:http|/)", minio_cmd)
    if endpts_found:
        minio_endpoint = endpts_found[0].rstrip('\x00').strip('\x00').strip(' ')
    else:
        logger.error("Unable to find MINIO Endpt from MINIO process")
        raise ValueError("Unable to find MINIO Endpt from MINIO process")
    return minio_endpoint


def get_minio_cluster_uuid(endpoint, url_prefix, cluster_id_url_suffix):
    url = str(url_prefix + endpoint
              + cluster_id_url_suffix)
    r = requests.get(url)
    minio_uuid = dict(r.json())["UUID"]
    return minio_uuid


def get_miniocluster_endpoint_map(mc_binary_path, logger):
    proc = subprocess.Popen(
        [mc_binary_path, 'config', 'host', 'list'],
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
        logger.error(f"Unable to read host list: {str(error)}")
        return {}

    if local_minio_host:
        conf_json_path = local_minio_host + "/dss/conf.json"
        try:
            proc = subprocess.Popen([mc_binary_path, 'cat', conf_json_path],
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
            logger.error(
                f"conf.json missing cluster information: {str(error)}"
            )
        except Exception as error:
            logger.error(
                f"Error when processing conf.json: {str(error)}")
    else:
        logger.error("Unable to find local minio host/endpoint")
        raise ValueError("Unable to find local minio host/endpoint")

    return miniocluster_endpoint_map
