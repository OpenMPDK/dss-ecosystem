#!/usr/bin/
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
from utils.utility import exception, exec_cmd, remoteExecution, get_s3_prefix, uploadFile
from utils.config import Config, commandLineArgumentParser, CommandLineArgument
from utils.signal_handler import SignalHandler

from logger import  MultiprocessingLogger

from multiprocessing import Process,Queue,Value, Lock, Manager
from worker import Worker
from monitor import Monitor
from nfs_cluster import NFSCluster
from task import Task, iterate_dir
import time
import zmq
import signal
from datetime import datetime
import json


manager = Manager()

class Master:
	"""
	Master application initiate
	-
	"""

	def __init__(self, operation, config):
		self.config = config
		self.client_ip_list = config.get("tess_clients_ip",[])
		self.workers_count = config["master"]["workers"]
		self.max_index_size = config["master"]["max_index_size"]
		self.workers = []
		self.clients = []
		self.task_queue = Queue()
		self.task_lock = Lock()

		self.operation=operation
		self.client_user_id = config["client"]["user_id"]
		self.client_password = config["client"]["password"]

		self.s3_config = config.get("s3_storage", {})

		## Logging
		self.logging_path = config.get("logging_path", "/var/log")
		self.logger       = None
		self.logger_queue = Queue()
		self.logger_lock  = Lock()

		self.lock = Lock()

		# Operation PUT/GET/DEL/LIST
		self.index_data_queue = Queue()  # Index data stored by workers
		self.index_data_lock = Lock()
		self.index_data_generation_complete = Value('i', 0)   ##TODO - Set value when all indexing is done.

		# Operation LIST
		self.prefix = config.get("prefix", None)
		self.listing_progress = manager.dict()
		self.listing_progress_lock = manager.Lock()

		# Status Progress
		self.index_data_count = Value('i', 0)  # File Index count shared between worker process.
		self.operation_start_time = None
		self.operation_end_time = None

		# Keep track of progress of hierarchical indexing.
		self.progress_of_indexing = manager.dict()
		self.progress_of_indexing_lock = manager.Lock()

		# Hierarchical indexing of leaf directory, to be used for listing.
		self.indexing_key = manager.dict()
		self.indexing_key = manager.Lock()

		# NFS shares
		self.nfs_shares = []




	def __del__(self):
		# Stop workers
		#self.stop_workers()
		# Stop Monitor
		# Stop Messaging
		# Stop clients
		#self.stop_logging()
		pass
		# Remove local NFS share


	def start(self):
		"""
		Start the master application with the following stuff
		- Start logger
		- Launch workers
		- Spawn clients
		- Start Monitor
		- Start Messaging ( File indexing message channel, progress status )
		:return:
		"""
		self.start_logging()
		self.start_workers()
		self.operation_start_time = datetime.now()
		if not self.operation.upper() == "LIST":
			self.spawn_clients()
			self.start_monitor()

		if self.operation.upper() == "PUT":
			self.start_indexing()
		if self.operation.upper() == "DEL" or self.operation.upper() == "GET":
			self.start_listing()

	def stop(self):
		"""
		Stop master and its all clients and their application.
		- Send notification to clients to stop
		- Stop all the workers
		- Stop Monitor
		- Stop messaging systems.
		:return:
		"""
		self.stop_workers()
		self.stop_clients()
		self.stop_monitor()
		self.nfs_cluster_obj.umount_all()
		self.stop_logging()

	def start_workers(self):
		"""
		Launch workers
		:return:
		"""
		index = 0
		while index < self.workers_count:
			w = Worker(id=index,
					   task_queue=self.task_queue,
					   logger_queue=self.logger_queue,
					   index_data_queue=self.index_data_queue,
					   progress_of_indexing= self.progress_of_indexing,
					   progress_of_indexing_lock=self.progress_of_indexing_lock,
					   listing_progress=self.listing_progress,
					   listing_progress_lock=self.listing_progress_lock,
					   s3_config=self.s3_config
					   )
			w.start()
			self.workers.append(w)
			index +=1

	def stop_workers(self, id=None):
		"""
		Stop worker/s
		:param id: worker id
		:return:
		"""
		index = 0
		while index < self.workers_count:
			w = self.workers[index]
			if w.get_status():
				w.stop()
			index += 1
		self.logger_queue.put("INFO: Stopped all the workers running with MasterApplication! ")
		print("INFO: Stopped all the workers running with MasterApplication! ")

	def spawn_clients(self):
		"""
		Spawn the clients
		:return:
		"""
		index =0
		for client_ip in self.client_ip_list:
			client = Client(index,
							client_ip,
							self.operation,
							self.logger_queue,
							self.config["master"]["ip_address"],
							self.client_user_id,
							self.client_password,
							self.config["dryrun"],
							self.config["message"],
							self.config.get("dest_path",""))
			client.start()
			self.logger_queue.put("INFO: Started ClientApplication-{} at node - {}".format(client.id,client_ip))
			self.clients.append(client)
			index +=1


	def stop_clients(self):
		"""
		Stop all the client application associated with this master application
		:return:
		"""
		for client in self.clients:
			client.stop()

	def start_monitor(self):
		"""
		Monitor the progress of the operation( PUT,DEL,LIST)
		:return:
		"""
		self.monitor = Monitor(self.clients,
							   self.index_data_queue,
							   self.index_data_lock,
							   self.index_data_generation_complete,
							   self.logger_queue,
							   self.logger_lock,
							   self.operation
							   )
		self.monitor.start()
	def stop_monitor(self):
		"""
		Stop all monitors forcefully.
		:return:
		"""
		if self.monitor:
			self.monitor.stop()

	def start_logging(self):
		"""
        Start Multiprocessing logger
        :return:
        """
		self.logger = MultiprocessingLogger(self.logging_path, __file__, self.logger_queue, self.logger_lock)
		self.logger.start()

	def stop_logging(self):
		"""
        Stop multiprocessing logger
        :return:
        """
		if not self.logger.status():
			self.logger.stop()


	def start_indexing(self):
		"""
		Indexing is required for PUT operation
		:return:
		"""
		# Fist Mount all NFS share locally
		self.nfs_cluster_obj = NFSCluster(self.config.get("nfs_config", {}), self.logger_queue)
		self.nfs_cluster_obj.mount_all()

		# Create first level task for each NFS share
		local_mounts = self.nfs_cluster_obj.get_mounts()
		#print(local_mounts)
		for ip_address, nfs_shares in local_mounts.items():
			#print("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
			self.logger_queue.put("INFO:NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
			self.nfs_shares.extend(nfs_shares)
			for nfs_share in nfs_shares:
				#print("DEBUG: Creating task for {}".format(nfs_share))
				task = Task(operation="indexing",
							data=nfs_share,
							nfs_cluster=ip_address,
							max_index_size=self.max_index_size)
				self.task_queue.put(task)


	def start_listing(self):

		print("INFO: Started Listing operation!")
		self.logger_queue.put("INFO: Started Listing operation!")
		# Create a Task based on prefix
		if self.prefix:
			print("DEBUG: Creating task for prefix - {}".format(self.prefix))
			self.prefix = get_s3_prefix(self.prefix)
			task= Task(operation="list",
					   data={"prefix": self.prefix},
					   s3config=self.config["s3_storage"],
					   max_index_size=self.config["master"].get("max_index_size", 10)
					   )
			self.task_queue.put(task)
		else:
			for ip_address, nfs_shares in self.config.get("nfs_config", {}).items():
				print("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
				self.logger_queue.put("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
				self.nfs_shares.extend(nfs_shares)
				for nfs_share in nfs_shares:
					print("DEBUG: Creating task for {}".format(nfs_share))
					prefix = get_s3_prefix(nfs_share)
					task = Task(operation="list",
								data={"prefix":prefix},
								s3config=self.config["s3_storage"],
								max_index_size=self.config["master"].get("max_index_size", 10)
								)
					self.task_queue.put(task)


	def compaction(self):
		# Spawn the process on target node and wait for response.
		command = "sudo python3 /usr/dss/nkv-datamover/target_compaction.py"
		compaction_status = {}
		start_time = datetime.now()
		for client_ip in self.config["dss_targets"]:
			print("INFO: Started Compaction for target-ip:{}".format(client_ip))
			self.logger_queue.put("INFO: Started Compaction for target-ip:{}".format(client_ip))
			ssh_client_handler, stdin, stdout, stderr = remoteExecution(client_ip, self.client_user_id, self.client_password,command)
			compaction_status[client_ip] = []
			compaction_status[client_ip].append(ssh_client_handler)
			compaction_status[client_ip].append(stdin)
			compaction_status[client_ip].append(stdout)
			compaction_status[client_ip].append(stderr)


		while True:
			is_compaction_done = True
			for client_ip in compaction_status:
				if compaction_status[client_ip][0]:
					if compaction_status[client_ip][2]:
						status = compaction_status[client_ip][2].channel.exit_status_ready()
						if status:
							print("INFO: Compaction is finished for - {}".format(client_ip))
							self.logger_queue.put("INFO: Compaction is finished for - {}".format(client_ip))
						else:
							is_compaction_done = False
			if is_compaction_done:
				break


		compaction_time = (datetime.now() - start_time).seconds
		print("INFO: Total Compaction time - {} seconds".format(compaction_time))







class Client:

	def __init__(self, id, ip, operation, logger_queue, master_ip, username="root", password="msl-ssg", dryrun=False, message_port={}, dest_path=""):
		self.id = id
		self.ip = ip
		self.operation=operation
		self.username = username
		self.password = password
		self.status = None
		self.ssh_client_handler = None
		self.master_ip_address = master_ip
		self.dryrun = dryrun
		self.destination_path = dest_path # Only to be used for GET operation

		# Messaging service configuration
		self.port_index = message_port["port_index"]  # Need to configure from configuration file.
		self.port_status = message_port["port_status"]
		self.socket_index = None
		self.socket_status = None

		# Remote output
		self.remote_execution_command =None
		self.remote_stdin =None
		self.remote_stdout = None
		self.remote_stderr = None

		# Execution status
		self.status = False

		# Logger
		self.logger_queue = logger_queue

	def __del__(self):
		"""
		Receive the PID of the remote process, so that forcefully remote process can be terminated.
		:return:
		"""
		if not self.status:
			self.stop()

	def start(self):
		"""
        Remote execution of client application ("client_application.py")
        :return:
        """
		# Setup
		#self.setup()
		print("INFO: Starting ClientApplication-{} on node {}".format(self.id,self.ip))
		self.logger_queue.put("INFO: Starting ClientApplication-{} on node {}".format(self.id,self.ip))
		command = "sudo python3 /usr/dss/nkv-datamover/client_application.py " + \
				                                                " --client_id {} ".format(self.id) + \
		                                                        " --operation {} ".format(self.operation) + \
		                                                        " --ip_address {} ".format(self.ip) + \
			                                                    " --port_index {} ".format(self.port_index) + \
			                                                    " --port_status {} ".format(self.port_status)

		if self.operation.upper() == "GET":
			command += " --dest_path {} ".format(self.destination_path)
		if self.master_ip_address == self.ip:
			command += " --master_node "
		if self.dryrun:
			command += " --dryrun "


		self.ssh_client_handler, stdin,stdout,stderr = remoteExecution(self.ip, self.username , self.password, command)
		self.remote_stdin = stdin
		self.remote_stdout = stdout
		self.remote_stderr = stderr

	def remote_client_status(self):
		if self.ssh_client_handler:
			if self.remote_stdout:
				self.status = self.remote_stdout.channel.exit_status_ready()
				if self.status:
					print("INFO: Remote ClientApplication-{} terminated!".format(self.id))
					self.logger_queue.put("INFO: Remote ClientApplication-{} terminated!".format(self.id))
				return self.status
		return False


	def remote_client_exit_status(self):
		"""
		It is blocking call.
		:return: exit status integer value
		"""
		if self.ssh_client_handler:

			exit_status = self.remote_stdout.channel.recv_exit_status()
			stdout_lines = self.remote_stdout.readlines()
			stderr_lines = self.remote_stderr.readlines()
			#print("INFO: Client-{} Remote execution status: {}".format(self.id,exit_status))
			self.logger_queue.put("INFO: Client-{} Remote execution status: {}".format(self.id,exit_status))
			if exit_status:
				print("DEBUG: Client-{} \n STDOUT {}".format(self.id,stdout_lines))
				self.logger_queue.put("DEBUG: Client-{} \n STDOUT {}".format(self.id,stdout_lines))
			if stderr_lines:
				print("ERROR: Client-{} \n STDERR {}".format(self.id,stderr_lines))
				self.logger_queue.put("ERROR: Client-{} \n STDERR {}".format(self.id,stderr_lines))
			self.ssh_client_handler.close()

		return exit_status


	def stop(self):
		"""
		Send a message to the client node through ZMQ.
		# TODO
		Need to terminate forcefully.
		:return:
		"""
		print("INFO: Stopping client-{}".format(self.id))
		if self.ssh_client_handler:

			status = self.remote_stdout.channel.recv_exit_status()
			stdout_lines = self.remote_stdout.readlines()
			stderr_lines = self.remote_stderr.readlines()
			print("INFO: Client-{} Remote execution status-{}".format(self.id,status))
			if status:
				print("DEBUG: Client-{} \n STDOUT {}".format(self.id,stdout_lines))
			if stderr_lines:
				print("ERROR: Client-{} \n STDERR {}".format(self.id,stderr_lines))
			self.ssh_client_handler.close()
			self.status = 0
		self.ssh_client_handler = None

	def setup(self):
		remote_dest_path = "/usr/test/datamover.tgz"
		username = "root"
		password = "msl-ssg"
		source_path = "/home/somnath.s/work/datamover.tgz"
		uploadFile(self.ip, remote_dest_path, source_path, username, password )

def process_put_operation(master):
	"""
	Manage the Upload process.

	Processes Stop Sequence:
	- Index-Monitor
	- Stop Workers if all index data distributed among client nodes.
	- Stop Status Poller
	- Stop Progress Tracer Monitor
	- Unmount remote NFS mounts.
	- Stop all clients
	:param master:
	:return:
	"""
	workers_stopped = 0
	unmounted_nfs_shares = 0
	monitors_stopped = 0

	client_applications_termination_waiting_message = True # Used when Workers and Monitors are stopped, but ClientApps.
	while True:
		# Check for completion of indexing, Shutdown workers
		indexing_done = True
		master.progress_of_indexing_lock.acquire()
		# len( ["/bird","/cat","/dog"] ) == len ( {"/bird":0, "/cat":0, "/dog":0} )
		if (len(master.nfs_shares) == len(master.progress_of_indexing)) and not workers_stopped:
			for nfs_share, status in master.progress_of_indexing.items():
				if status:
					indexing_done = False

			## Check if index generation is completed by the worker processes.
			if indexing_done and master.index_data_generation_complete.value == 0:
				# Shut down Monitor-Index at Master
				master.index_data_lock.acquire()
				master.index_data_generation_complete.value = 1
				master.index_data_lock.release()

				master.logger_queue.put("INFO: Indexed data generation is completed!")
				print("INFO: Indexed data generation is completed!")
				print("INFO: {} INDEXING Completed in {} seconds".format(master.operation,(datetime.now() - master.operation_start_time).seconds))

		master.progress_of_indexing_lock.release()

		# Check all the ClientApplications once they are finished
		all_clients_completed = 1
		for client in master.clients:
			#print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
			if not client.status:
				if client.remote_client_status() :
					client.remote_client_exit_status()
				all_clients_completed = 0

		if all_clients_completed:
			master.logger_queue.put("INFO: All ClientApplications terminated gracefully !") ## Termination3
		elif  workers_stopped and monitors_stopped:
			if client_applications_termination_waiting_message:
				print("INFO: Waiting for ClientApplication to stop!")
				client_applications_termination_waiting_message = False

		# Check for Monitors status
		master.monitor.status_lock.acquire()
		if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
				master.monitor.monitor_status_poller.value and \
				master.monitor.monitor_progress_status.value:
			monitors_stopped = 1
		master.monitor.status_lock.release()

		# Un-mount device once all monitors are stopped. Because, if ClientApp is launches at the same node of master, then
		# un-mount should not happen.
		if monitors_stopped and  not unmounted_nfs_shares:
			print("INFO: Un-mount all NFS shares at Master")
			master.logger_queue.put("INFO: Un-mount all NFS shares at Master")
			master.nfs_cluster_obj.umount_all()  ## Termination2
			unmounted_nfs_shares = 1


		# Bring down workers.
		if not workers_stopped  and master.monitor.monitor_index_data_sender.value:
			master.stop_workers()
			workers_stopped = 1

		## Once all the response received from Client Applications, shut down 3 monitors
		if workers_stopped and monitors_stopped and all_clients_completed:
			break

		# If ClientApplications running on client nodes gets terminated, then shutdown workers, monitors forcefully to exit program
		if all_clients_completed:
			if not monitors_stopped:
				master.stop_monitor()
			if not workers_stopped:
				master.stop_workers()
			break

		time.sleep(2)



def process_list_operation(master):
	master.start_listing()
	while True:
		try:
			if master.prefix and master.prefix in master.listing_progress and master.listing_progress[master.prefix] == 0:
				break
			elif len(master.nfs_shares) == len(master.listing_progress):
				listing_completed = True
				for prefix,prefix_processing_status in master.listing_progress.items():
					if prefix_processing_status > 0:
						listing_completed = False
				if listing_completed:
					break
		except Exception as e:
			print("EXECPTION: Listing - {}".format(e))
		time.sleep(1)
	master.stop_workers()
	print("LISTING:{}".format(master.listing_progress))


def process_del_operation(master):
	workers_stopped = 0
	monitors_stopped = 0

	while True:
		# Check for completion of indexing, Shutdown workers
		listing_done = False
		master.listing_progress_lock.acquire()

		# Determine listing is done.
		if master.prefix:
			if len(master.listing_progress) == 1 and master.listing_progress.get(master.prefix, -1) == 0:
				listing_done = True

		else:
			# All the children of top level prefix should be processed.
			# len( ["/bird","/cat","/dog"] ) == len ( {"bird/":0, "cat/":0, "dog/":0} )
			if (len(master.nfs_shares) == len(master.listing_progress)):
				is_all_child_processed = True
				for nfs_share, child_listing_count in master.listing_progress.items():
					if child_listing_count :
						is_all_child_processed = False

				listing_done = is_all_child_processed
		master.listing_progress_lock.release()
		#print("NFS Share-{}:{}:{}".format(master.nfs_shares, master.listing_progress, listing_done))

		if not workers_stopped:
			if listing_done and master.index_data_generation_complete.value == 0:
				master.index_data_generation_complete.value = 1
				master.logger_queue.put("INFO: Object-Keys generation through listing is completed!")
				print("INFO: Object-Keys generation through listing is completed!")
				print("INFO: {} LISTING Completed in {} seconds".format(master.operation, (
						datetime.now() - master.operation_start_time).seconds))
				# Shutdown workers
				#master.stop_workers()
				#workers_stopped = 1


		# Check all the ClientApplications once they are finished
		all_clients_completed = 1
		for client in master.clients:
			#print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
			if not client.status:
				if client.remote_client_status() :
					client.remote_client_exit_status()
				all_clients_completed = 0

		# Check for Monitors status
		master.monitor.status_lock.acquire()
		if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
				master.monitor.monitor_status_poller.value and \
				master.monitor.monitor_progress_status.value:
			monitors_stopped = 1
		master.monitor.status_lock.release()

		# Bring down workers.
		if not workers_stopped and master.monitor.monitor_index_data_sender.value:
			master.stop_workers()
			workers_stopped = 1

		## Once all the response received from Client Applications, shut down 3 monitors
		if workers_stopped and monitors_stopped and all_clients_completed:
			break

		# If ClientApplications running on client nodes gets terminated, then shutdown workers, monitors forcefully to exit program
		if all_clients_completed:
			if not monitors_stopped:
				master.stop_monitor()
			if not workers_stopped:
				master.stop_workers()
			break

		time.sleep(2)

def process_get_operation(master):
	workers_stopped = 0
	monitors_stopped = 0

	while True:
		# Check for completion of indexing, Shutdown workers
		listing_done = False
		master.listing_progress_lock.acquire()

		# Determine listing is done.
		if master.prefix:
			if len(master.listing_progress) == 1 and master.listing_progress.get(master.prefix, -1) == 0:
				listing_done = True

		else:
			# All the children of top level prefix should be processed.
			# len( ["/bird","/cat","/dog"] ) == len ( {"bird/":0, "cat/":0, "dog/":0} )
			if (len(master.nfs_shares) == len(master.listing_progress)):
				is_all_child_processed = True
				for nfs_share, child_listing_count in master.listing_progress.items():
					if child_listing_count:
						is_all_child_processed = False

				listing_done = is_all_child_processed
		master.listing_progress_lock.release()
		# print("NFS Share-{}:{}:{}".format(master.nfs_shares, master.listing_progress, listing_done))

		if not workers_stopped:
			if listing_done and master.index_data_generation_complete.value == 0:
				master.index_data_generation_complete.value = 1
				master.logger_queue.put("INFO: Object-Keys generation through listing is completed!")
				print("INFO: Object-Keys generation through listing is completed!")
				print("INFO: {} LISTING Completed in {} seconds".format(master.operation, (
						datetime.now() - master.operation_start_time).seconds))
		# Shutdown workers
		# master.stop_workers()
		# workers_stopped = 1

		# Check all the ClientApplications once they are finished
		all_clients_completed = 1
		for client in master.clients:
			# print("Client Status:{}, Client_id-{}".format(client.remote_client_status(),client.id))
			if not client.status:
				if client.remote_client_status():
					client.remote_client_exit_status()
				all_clients_completed = 0

		# Check for Monitors status
		master.monitor.status_lock.acquire()
		if not monitors_stopped and master.monitor.monitor_index_data_sender.value and \
				master.monitor.monitor_status_poller.value and \
				master.monitor.monitor_progress_status.value:
			monitors_stopped = 1
		master.monitor.status_lock.release()

		# Bring down workers.
		if not workers_stopped and master.monitor.monitor_index_data_sender.value:
			master.stop_workers()
			workers_stopped = 1

		## Once all the response received from Client Applications, shut down 3 monitors
		if workers_stopped and monitors_stopped and all_clients_completed:
			break

		# If ClientApplications running on client nodes gets terminated, then shutdown workers, monitors forcefully to exit program
		if all_clients_completed:
			if not monitors_stopped:
				master.stop_monitor()
			if not workers_stopped:
				master.stop_workers()
			break

		time.sleep(1)


if __name__ == "__main__":
	cli = CommandLineArgument()
	print("INFO: Performing {} operation".format(cli.operation))
	#print(cli.options)
	# operation, params = commandLineArgumentParser()

	# Add signal handler
	#signal_handler = SignalHandler()
	#signal_handler.initiate()


	params = cli.options
	config_obj = Config(params)
	config = config_obj.get_config()
	print(config)

	master = Master(cli.operation, config)
	now = datetime.now()
	print("INFO:Starting master!")
	master.start()

	#signal_handler.registered_functions.append(master.nfs_cluster_obj.umount_all)

	if cli.operation.upper() == "PUT":
		process_put_operation(master)
	elif cli.operation.upper() == "LIST":
                print('Unsupported operation')
		# process_list_operation(master)
	elif cli.operation.upper() == "DEL":
		process_del_operation(master)
	elif cli.operation.upper() == "GET":
		process_get_operation(master)

	# Start Compaction
	if cli.operation == "PUT" and "compaction" in params and params["compaction"]:
		master.compaction()

	# Terminate logger at the end.
	master.stop_logging()  ## Termination5
	master.nfs_cluster_obj.umount_all()
	print("INFO: Stopping master")
	### 10.1.51.238 , 10.1.51.54, 10.1.51.61



