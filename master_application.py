#!/usr/bin/python
import os,sys
from utils.utility import exception, exec_cmd, remoteExecution
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


manager = Manager()

class Master:
	"""
	TODO:
	-
	"""

	def __init__(self, config):
		self.config = config
		self.client_ip_list = config.get("tess_clients_ip",[])
		self.workers_count = config["master"]["workers"]
		self.max_index_size = config["master"]["max_index_size"]
		self.workers = []
		self.clients = []
		self.task_queue = Queue()
		self.task_lock = Lock()



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

		# Status Progress
		self.index_data_count = Value('i', 0)  # File Index count shared between worker process.

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
		self.stop_workers()
		# Stop Monitor
		# Stop Messaging
		# Stop clients
		self.stop_logging()


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
		self.start_indexing()

		self.spawn_clients()


		# Test Data - Index
		#self.index_data_queue.put(["f1","d1","d2","d3","d4","d5"])
		#self.index_data_queue.put(["f2", "f3"])
		#self.index_data_queue.put(["f4", "f5", "f6"])

		self.start_monitor()



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
					   progress_of_indexing_lock=self.progress_of_indexing_lock
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
			client = Client(index, client_ip)
			#print("INFO: Starting client application at node - {}".format(client_ip))
			client.start()
			self.logger_queue.put("INFO: Started Client Application-{} at node - {}".format(client.id,client_ip))
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
							   self.logger_lock
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
		#print("NFS Mounting!!")
		# Fist Mount all NFS share locally
		self.nfs_cluster_obj = NFSCluster(self.config.get("nfs_config", {}), self.logger_queue)
		self.nfs_cluster_obj.mount_all()

		# Create first level task for each NFS share
		local_mounts = self.nfs_cluster_obj.get_mounts()
		#print(local_mounts)
		for ip_address, nfs_shares in local_mounts.items():
			print("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
			self.logger_queue.put("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
			self.nfs_shares.extend(nfs_shares)
			for nfs_share in nfs_shares:
				print("DEBUG: Creating task for {}".format(nfs_share))
				task = Task(operation="indexing",
							data=nfs_share,
							nfs_cluster=ip_address,
							max_index_size=self.max_index_size)
				self.task_queue.put(task)


class Client:

	def __init__(self, id, ip, username="root", password="msl-ssg"):
		self.id = id
		self.ip = ip
		self.username = username
		self.password = password
		self.status = None
		self.ssh_client_handler = None

		# Messaging service configuration
		self.port_index = "6000"  # Need to configure from configuration file.
		self.port_status = "6001"
		self.socket_index = None
		self.socket_status = None

		# Remote output
		self.remote_execution_command =None
		self.remote_stdin =None
		self.remote_stdout = None
		self.remote_stderr = None

		# Execution status
		self.status = 0

	def __del__(self):
		"""
		Receive the PID of the remote process, so that forcefully remote process can be terminated.
		:return:
		"""
		if self.status:
			self.stop()

	def start(self):
		"""
        Remote execution of client application ("client_application.py")
        :return:
        """
		print("INFO: Starting Client Application-{} on node {}".format(self.id,self.ip))
		command = "python3 /home/somnath.s/work/nkv-datamover/client_application.py -id {}".format(self.id)
		self.ssh_client_handler, stdin,stdout,stderr = remoteExecution(self.ip, self.username , self.password, command)
		self.remote_stdin = stdin
		self.remote_stdout = stdout
		self.remote_stderr = stderr
		self.status = 1


	def stop(self):
		"""
		Send a message to the client node through ZMQ.
		# TODO
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


if __name__ == "__main__":
	cli = CommandLineArgument()
	print(cli.operation)
	print(cli.options)
	# operation, params = commandLineArgumentParser()

	# Add signal handler
	signal_handler = SignalHandler()
	signal_handler.initiate()


	params = cli.options
	config_obj = Config(params)
	config = config_obj.get_config()

	master = Master(config)
	now = datetime.now()
	print("Starting master! Time-{}".format(now.strftime("%H:%M:%S")))
	master.start()

	signal_handler.registered_functions.append(master.nfs_cluster_obj.umount_all)
	#time.sleep(60)
	#master.stop()

	workers_stopped = 0
	unmounted_nfs_shares = 0
	monitors_stopped = 0

	# Crate a LOOP
	# Make sure indexing is completed: Total no of NFS share == Number of indexing root

	# Then stop all workers which is idle only.
	#
	while True:
		# Check for completion of indexing, Shutdown workers
		indexing_done = True
		master.progress_of_indexing_lock.acquire()
		# len( ["/bird","/cat","/dog"] ) == len ( {"/bird":0, "/cat":0, "/dog":0} )
		if ( len(master.nfs_shares) == len(master.progress_of_indexing) ) and not workers_stopped:
			for nfs_share, status in master.progress_of_indexing.items():
				if status:
					indexing_done = False

			## Stop workers
			if indexing_done:
				# Shut down Monitor-Index at Master
				master.index_data_lock.acquire()
				master.index_data_generation_complete.value = 1
				master.index_data_lock.release()

				master.logger_queue.put("INFO: Indexed data distribution is completed!")
				print("INFO: Indexed data distribution is completed!")
				master.stop_workers() ## Termination1
				workers_stopped = 1

				print("INFO: Un-mount all NFS shares at Master")
				master.logger_queue.put("INFO: Un-mount all NFS shares at Master")
				master.nfs_cluster_obj.umount_all() ## Termination2
				unmounted_nfs_shares = 1

		master.progress_of_indexing_lock.release()

		# Check all the ClientApplications once they are finished
		all_clients_completed = 1
		"""
		for client in master.clients:
			status = client.remote_stdout.channel.recv_exit_status()
			if status == 0:
				client.status = 0
			else:
				all_clients_completed = 0

		if all_clients_completed:
			master.logger_queue.put("INFO: All ClientApplications terminated gracefully !") ## Termination3
		"""

		# Check for Monitors status
		master.monitor.status_lock.acquire()
		if master.monitor.monitor_index_data_sender.value and \
			master.monitor.monitor_status_poller.value and \
			master.monitor.monitor_progress_status.value :
			monitors_stopped = 1
		master.monitor.status_lock.release()
		## Once all the response received from Client Applications, shut down 3 monitors
		if workers_stopped and all_clients_completed and monitors_stopped:
			break

		time.sleep(2)


	# Terminate logger at the end.
	master.stop_logging()  ## Termination5
	print("Stopping master")
	### 10.1.51.238 , 10.1.51.54, 10.1.51.61
