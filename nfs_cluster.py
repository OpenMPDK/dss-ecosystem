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
    def __init__(self, config={}, logger_queue=None):
        self.config = config
        self.local_mounts = {}
        self.nfs_cluster = []
        self.mounted = False
        self.logger_queue = logger_queue


    def __del__(self):
        # Unmount all the mounted local NFS paths.
        if self.mounted:
            for cluster_ip in self.local_mounts:
                for local_mount in self.local_mounts[cluster_ip]:
                    command = "sudo umount {}".format(local_mount)
                    ret, console = exec_cmd(command, True, True)
                    if ret :
                        print("ERROR: Failed to un-mount  {} path ".format(local_mount))
                        self.logger_queue.put("ERROR: Failed to un-mount  {} path ".format(local_mount))
                    else:
                        self.logger_queue.put("DEBUG: Un-mounted local NFS mount {}".format(local_mount))
            env.close()

    @exception
    def mount_all(self):
        """
        Mount all the NFS shares from each cluster specified in configuration file.
        :return: None
        """
        if self.config:
            # perform mounting for each cluster
            for cluster_ip in self.config:
                local_nfs_mount_paths = []
                mounted_nfs_shares = []
                for nfs_share in self.config[cluster_ip]:
                    ret,console = self.mount(cluster_ip, nfs_share)
                    if ret == 0:
                        mounted_nfs_shares.append(nfs_share)
                        local_nfs_mount_paths.append(nfs_share)
                    else:
                        print("ERROR:NFS mounting {}:{} failed \n {}".format(cluster_ip,nfs_share, console))
                        self.logger_queue.put("ERROR:NFS mounting {}:{} failed \n {}".format(cluster_ip,nfs_share, console))

                print("INFO: Mounted NFS shares {}:{}".format(cluster_ip, mounted_nfs_shares))
                self.logger_queue.put("INFO: Mounted NFS shares {}:{}".format(cluster_ip, mounted_nfs_shares))
                if local_nfs_mount_paths:
                    self.local_mounts[cluster_ip] = local_nfs_mount_paths
                    self.nfs_cluster.append(cluster_ip)

                self.mounted = True

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
        ret =0
        console =None
        # Create local directory
        if not os.path.isdir(nfs_share):
            os.mkdir(nfs_share)
        if os.path.ismount(nfs_share):
            print("WARNING: NFS share \"{}\" already mounted!".format(nfs_share))
            self.logger_queue.put("WARNING: NFS share {} already mounted!".format(nfs_share))
        else:
            command = "sudo mount {}:{} {}".format(cluster_ip, nfs_share, nfs_share)
            #self.logger_queue.put("DEBUG: Mount Command - {}".format(command))
            ret, console = exec_cmd(command, True, True)
        return ret,console

    @exception
    def umount_all(self):
        """
        Un-mount all the mounted local NFS paths.
        :return:
        """
        for cluster_ip in self.local_mounts:
            local_mounts = []
            self.logger_queue.put("{}:{}".format(cluster_ip, self.local_mounts[cluster_ip]))
            for local_mount in self.local_mounts[cluster_ip]:
                ret, console = self.umount(local_mount)
                if ret:
                    print("ERROR: Failed to un-mount  {} path ".format(local_mount))
                    self.logger_queue.put("ERROR: Failed to un-mount  {} path ".format(local_mount))
                else:
                    #print("DEBUG: Un-mounted local NFS mount {}".format(local_mount))
                    #self.logger_queue.put("DEBUG: Un-mounted local NFS mount {}".format(local_mount))
                    local_mounts.append(local_mount)

            print("INFO: Un-mounted local NFS mount => NFS Cluster-{}:{}".format(cluster_ip, local_mounts))
            self.logger_queue.put("INFO: Un-mounted local NFS mount => NFS Cluster-{}:{}".format(cluster_ip, local_mounts))

        self.mounted = False

    @exception
    def umount(self, local_mount):

        ret =0
        console =None
        if os.path.ismount(local_mount):
            command = "sudo umount {}".format(local_mount)
            ret, console = exec_cmd(command, True, True)
            # Remove directory
            if ret == 0:
                try:
                    os.rmdir(local_mount)
                except OSError as e:
                    self.logger_queue.put("ERROR: Unable to remove directory {}, {} ".format(local_mount, e))
        else:
            self.logger_queue.put("WARNING: {} path is not a mounted path".format(local_mount))
        return ret,console

    def get_cluster(self):
        return self.nfs_cluster

    def get_mounts(self):
        return self.local_mounts


