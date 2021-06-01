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
import os,sys
from utils.utility import exception, exec_cmd
from multiprocessing import Manager


class NFSCluster:
    manager = Manager()
    def __init__(self, config={}, logger=None):
        self.config = config
        self.local_mounts = {}
        self.nfs_cluster = []
        self.mounted = False
        self.logger = logger


    def __del__(self):
        # Unmount all the mounted local NFS paths.
        if self.mounted:
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
                mounted_nfs_shares = []
                for nfs_share in self.config[cluster_ip]:
                    nfs_share = os.path.abspath(nfs_share)
                    ret,console = self.mount(cluster_ip, nfs_share)
                    if ret == 0:
                        mounted_nfs_shares.append(nfs_share)

                if mounted_nfs_shares:
                    self.logger.info("Mounted NFS shares {}:{}".format(cluster_ip, mounted_nfs_shares))
                    self.nfs_cluster.append(cluster_ip)

    @exception
    def mount(self,cluster_ip, nfs_share):
        """
        Mount a NFS share from a cluster.
        :param cluster_ip: NFS cluster IP (Required)
        :param nfs_share: NFS share (Required) ex: /data
        :return:
        ret : a integer value. 0 to indicate success.
        console: STDOUT message
        """
        ret = None
        console =None
        # Generate a unique md5sum hash key for NFS shares
        nfs_share_mount = os.path.abspath("/" + cluster_ip + "/" + nfs_share)
        nfs_share_already_mounted = False

        # Create local directory
        if not os.path.isdir(nfs_share_mount):
            # os.mkdir(nfs_share)
            command = "mkdir -p {}".format(nfs_share_mount)
            ret, console = exec_cmd(command, True, True)
        if os.path.ismount(nfs_share_mount):
            self.logger.warn("NFS share {} already mounted to {}".format(nfs_share, nfs_share_mount))
            nfs_share_already_mounted = True
        else:
            command = "mount {}:{} {}".format(cluster_ip, nfs_share, nfs_share_mount)
            ret, console = exec_cmd(command, True, True)

        if ret == 0:
            self.logger.info("NFS mounting {}:{} => {} successful".format(cluster_ip, nfs_share, nfs_share_mount))
        elif ret:
            self.logger.error("NFS mounting {}:{} failed \n {}".format(cluster_ip, nfs_share, console))

        if ret == 0 or nfs_share_already_mounted:
            if cluster_ip not in self.local_mounts:
                self.local_mounts[cluster_ip] = [nfs_share]
            else:
                self.local_mounts[cluster_ip].append(nfs_share)
            self.mounted = True
        return ret,console

    @exception
    def umount_all(self):
        """
        Un-mount all the mounted local NFS paths.
        :return:
        """
        for cluster_ip in self.local_mounts:
            nfs_mounts = []
            for nfs_share in self.local_mounts[cluster_ip]:
                local_nfs_mount = os.path.abspath("/" + cluster_ip + "/" + nfs_share)
                ret, console = self.umount(local_nfs_mount)
                if ret == 0:
                    nfs_mounts.append(nfs_share)
            if nfs_mounts:
                self.logger.info("Un-mounted local NFS mount => NFS Cluster-{}:{}".format(cluster_ip, nfs_mounts))
            else:
                self.logger.info("Un-mount failed for local NFS mount => NFS Cluster-{}:{}".format(cluster_ip, nfs_mounts))
        self.mounted = False

    @exception
    def umount(self, local_mount):

        ret = None
        console =None
        if os.path.ismount(local_mount):
            command = "umount {}".format(local_mount)
            ret, console = exec_cmd(command, True, True)
            # Remove directory
            if ret == 0:
                try:
                    command = "rm -rf {}".format(local_mount)
                    ret, console = exec_cmd(command, True, True)
                    if ret:
                        self.logger.error("Failed to remove {} ".format(local_mount))
                except OSError as e:
                    self.logger.error("Unable to remove directory {}, {} ".format(local_mount, e))
            else:
                self.logger.error("Failed to un-mount {} path ".format(local_mount))
        else:
            self.logger.warn("{} path is not a mounted path".format(local_mount))
        return ret,console

    def get_cluster(self):
        return self.nfs_cluster

    def get_mounts(self):
        return self.local_mounts


