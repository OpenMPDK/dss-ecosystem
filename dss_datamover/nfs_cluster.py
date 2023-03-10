#!/usr/bin/python

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
from utils.utility import exception, exec_cmd, first_delimiter_index
from multiprocessing import Manager


class NFSCluster:
    manager = Manager()

    def __init__(self, config={}, user_id="ansible", password="password", logger=None):
        self.config = config.get("nfs", {})
        self.local_mounts = {}
        self.mounted_nfs_shares = []
        self.nfs_cluster = []
        self.mounted = config.get("mounted", False)
        self.logger = logger
        self.user_id = user_id
        self.password = password

    def __del__(self):
        # Unmount all the mounted local NFS paths.
        if self.mounted and self.mounted_nfs_shares:
            self.umount_all()

    @exception
    def mount_all(self):
        """
        Mount all the NFS shares from each cluster specified in configuration file.
        :return: None
        """
        if self.config:
            # perform mounting for each cluster
            for cluster_ip in self.config:
                mounted_nfs_shares_cluster_ip = []
                for nfs_share in self.config[cluster_ip]:
                    nfs_share = os.path.abspath(nfs_share)
                    ret, console = self.mount(cluster_ip, nfs_share)
                    if ret == 0:
                        mounted_nfs_shares_cluster_ip.append(nfs_share)

                if mounted_nfs_shares_cluster_ip:
                    self.logger.info("Mounted NFS shares {}:{}".format(cluster_ip, mounted_nfs_shares_cluster_ip))
                    self.nfs_cluster.append(cluster_ip)

    @exception
    def mount_based_on_prefix(self, prefix):
        """
        Mount only the specified prefix and store cluster_ip into nfs_cluster list.
        :param prefix:
        :return: touple => ( cluster_ip, mounted_path , return code)
        """
        first_delimiter_pos = first_delimiter_index(prefix, "/")
        cluster_ip = prefix[0:first_delimiter_pos]
        ret = -1
        nfs_share = ""
        console = ""
        for nfs_share in self.config[cluster_ip]:
            nfs_share_prefix = cluster_ip + nfs_share
            if prefix.startswith(nfs_share_prefix):
                if (cluster_ip in self.local_mounts
                        and nfs_share in self.local_mounts[cluster_ip]):
                    self.logger.info("Prefix -{} is already mounted to {}".format(
                        prefix, "/" + nfs_share_prefix))
                    return cluster_ip, nfs_share, 0, console
                else:
                    ret, console = self.mount(cluster_ip, nfs_share)
                break
        if ret == 0:
            self.logger.info("Mounted NFS shares {}:{}".format(cluster_ip, nfs_share))
            self.nfs_cluster.append(cluster_ip)

        return cluster_ip, nfs_share, ret, console

    @exception
    def mount(self, cluster_ip, nfs_share):
        """
        Mount a NFS share from a cluster.
        :param cluster_ip: NFS cluster IP (Required)
        :param nfs_share: NFS share (Required) ex: /data
        :return:
        ret : a integer value. 0 to indicate success.
        console: STDOUT message
        """
        ret = None
        console = None
        # Generate a unique md5sum hash key for NFS shares
        nfs_share_mount = os.path.abspath("/" + cluster_ip + "/" + nfs_share)
        nfs_share_already_mounted = False

        # Already File System path is mounted
        if self.mounted:
            if os.path.isdir(nfs_share):
                nfs_share_already_mounted = True
            else:
                self.logger.error("FS path {} doesn't exist ".format(nfs_share))
        else:
            if os.path.ismount(nfs_share_mount):
                nfs_share_already_mounted = True
                self.logger.warn("NFS share {} already mounted to {}".format(nfs_share, nfs_share_mount))
            elif not os.path.isdir(nfs_share_mount):
                command = "mkdir -p {}".format(nfs_share_mount)
                dir_ret, console = exec_cmd(command, True, True, self.user_id)
                if dir_ret:
                    self.logger.fatal("Faild to create the directory {} for mount".format(nfs_share_mount))
                    return dir_ret, console
            # Mount FS
            if not nfs_share_already_mounted:
                command = "mount {}:{} {}".format(cluster_ip, nfs_share, nfs_share_mount)
                ret, console = exec_cmd(command, True, True, self.user_id)

                if ret == 0:
                    self.mounted_nfs_shares.append(nfs_share)
                    self.logger.info("NFS mounting {}:{} => {} successful".format(
                        cluster_ip, nfs_share, nfs_share_mount))
                elif ret:
                    self.logger.error("NFS mounting {}:{} failed \n {}".format(
                        cluster_ip, nfs_share, console))

        if ret == 0 or nfs_share_already_mounted:
            if cluster_ip not in self.local_mounts:
                self.local_mounts[cluster_ip] = [nfs_share]
            else:
                self.local_mounts[cluster_ip].append(nfs_share)
            ret = 0  # Fix issue MIN-1238

        return ret, console

    @exception
    def umount_all(self):
        """
        Un-mount all the mounted local NFS paths.
        :return:
        """
        for cluster_ip in self.local_mounts:
            nfs_unmount_success = []
            nfs_unmount_failed = []
            for nfs_share in self.local_mounts[cluster_ip]:
                if nfs_share in self.mounted_nfs_shares:
                    local_nfs_mount = os.path.abspath("/" + cluster_ip + "/" + nfs_share)
                    ret, console = self.umount(local_nfs_mount)
                    if ret == 0:
                        nfs_unmount_success.append(nfs_share)
                    else:
                        nfs_unmount_failed.append(nfs_share)
            if self.mounted_nfs_shares:
                if nfs_unmount_success:
                    self.logger.info("Un-mounted local NFS mount => NFS Cluster-{}:{}".format(
                        cluster_ip, nfs_unmount_success))
                if nfs_unmount_failed:
                    self.logger.info("Un-mount failed for local NFS mount => NFS Cluster-{}:{}".format(
                        cluster_ip, nfs_unmount_failed))
        self.mounted = False

    @exception
    def umount(self, local_mount):

        ret = None
        console = None
        if os.path.ismount(local_mount):
            command = "umount {}".format(local_mount)
            ret, console = exec_cmd(command, True, True, self.user_id)
            # Remove directory
            if ret == 0:
                try:
                    command = "rm -rf {}".format(local_mount)
                    ret, console = exec_cmd(command, True, True, self.user_id)
                    if ret:
                        self.logger.error("Failed to remove {} ".format(local_mount))
                except OSError as e:
                    self.logger.error("Unable to remove directory {}, {} ".format(local_mount, e))
            else:
                self.logger.error("Failed to un-mount {} path ".format(local_mount))
        else:
            self.logger.warn("{} path is not a mounted path".format(local_mount))

        return ret, console

    def get_cluster(self):
        return self.nfs_cluster

    def get_mounts(self):
        return self.local_mounts
