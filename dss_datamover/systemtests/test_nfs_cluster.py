#!/usr/bin/python3

"""
# The Clear BSD License
#
# Copyright (c) 2023 Samsung Electronics Co., Ltd.
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
import pytest


@pytest.mark.usefixtures(
   "get_nfs_cluster",
   "get_system_config_dict",
   "generate_full_ip_prefix"
)
class TestNFSCluster:
    """
    Used for NFS cluster related system tests.
    """

    def test_nfs_mount_all(self, get_nfs_cluster, get_system_config_dict):
        get_nfs_cluster.mount_all()
        for cluster_ip in get_system_config_dict["nfs"]:
            for nfs_share in get_system_config_dict["nfs"][cluster_ip]:
                nfs_share_mount = os.path.abspath("/" + cluster_ip + "/" + nfs_share) if \
                get_nfs_cluster.server_as_prefix else os.path.abspath("/" + nfs_share)
                assert os.path.exists(nfs_share_mount)

    def test_nfs_mount_based_on_prefix(self, get_nfs_cluster, get_system_config_dict, generate_full_ip_prefix):
        for ip, nfs_shares in get_system_config_dict["fs_config"]["nfs"].items():
            prefix = nfs_shares[0]
            break
        get_nfs_cluster.mount_based_on_prefix(prefix)
        nfs_share_mount = os.path.abspath("/" + cluster_ip + "/" + prefix) if \
                get_nfs_cluster.server_as_prefix else os.path.abspath("/" + prefix)
        assert os.path.exists(nfs_share_mount)
