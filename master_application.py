#!/usr/bin/python
import os,sys
from utils.utility import exception, exec_cmd, remoteExecution
from utils.config import Config, commandLineArgumentParser, CommandLineArgument

from logger import  MultiprocessingLogger

from multiprocessing import Process,Queue,Value, Lock
from worker import Worker
from monitor import Monitor
from nfs_cluster import NFSCluster
from task import Task, iterate_dir
import time
import zmq


class MasterSharedResource:

	# All the shared resource used by all components at ClientApplication

	def __init__(self):
		self.task_queue = Queue()
		self.index_queue = Queue()

class Master:
	"""
	TODO:
	-
	"""

	def __init__(self, config):
		self.config = config
		self.client_ip_list = config.get("tess_clients_ip",[])
		self.workers_count = config["master"]["workers"]
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
		## Clients

		## Workers
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
					   index_data_queue=self.index_data_queue)
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

	def spawn_clients(self):
		"""
		Spawn the clients
		:return:
		"""
		index =0
		for client_ip in self.client_ip_list:
			client = Client(index, client_ip)
			print("INFO: Starting client application at node - {}".format(client_ip))
			client.start()
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
		print("NFS Mounting!!")
		# Fist Mount all NFS share locally
		self.nfs_cluster_obj = NFSCluster(self.config.get("nfs_config", {}), self.logger_queue)
		self.nfs_cluster_obj.mount_all()

		# Create first level task for each NFS share
		local_mounts = self.nfs_cluster_obj.get_mounts()
		#print(local_mounts)
		for ip_address, nfs_shares in local_mounts.items():
			print("NFS Cluster:{}, NFS Shares:{}".format(ip_address, nfs_shares))
			for nfs_share in nfs_shares:
				print("DEBUG: Creating task for {}".format(nfs_share))
				task = Task(operation="indexing",
							data=nfs_share,
							nfs_cluster=ip_address)
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
		print("INFO: Starting client-{} on node {}".format(self.id,self.ip))
		command = "python3 /home/somnath.s/work/FB_Data_Mover/AmazonSDK/client_application.py -id {}".format(self.id)
		self.ssh_client_handler, stdin,stdout,stderr = remoteExecution(self.ip, self.username , self.password, command)
		self.remote_stdin = stdin
		self.remote_stdout = stdout
		self.remote_stderr = stderr

		# Debug client side output log
		#stdout,stderr,status = remoteExecution(self.ip, self.username, self.password, command, True)
		#print("STDOUT:{}\n STDERR:{}\n STATUS:{}".format(stdout,stderr,status))

		# Create 2 sockets for the client to communicate to ClientApplication


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
			if status == 0:
				print("DEBUG: Client-{} \n STDOUT {}".format(self.id,stdout_lines))
			if stderr_lines:
				print("ERROR: Client-{} \n STDERR {}".format(self.id,stderr_lines))
			self.ssh_client_handler.close()

		"""
		command = "sudo pkill client_application.py "
		stdout,stderr,status = remoteExecution(self.ip, self.username, self.password, command, True)
		print("{}:{}:{}".format(stdout,stderr,status))
		self.status = 0
		if status == 0:
			# Client application got terminated gracefully
			self.status = 0
		else:
			print("ERROR: {}".format(stderr))
		"""


if __name__ == "__main__":
	cli = CommandLineArgument()
	print(cli.operation)
	print(cli.options)
	# operation, params = commandLineArgumentParser()

	params = cli.options
	config_obj = Config(params)
	config = config_obj.get_config()

	master = Master(config)
	print("Starting master")
	master.start()
	time.sleep(30)
	print("Stopping master")
	master.stop()


	### 10.1.51.238 , 10.1.51.54, 10.1.51.61
