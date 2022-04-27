import sys
import time
from logger import MultiprocessingLogger
from utils.config import Config, ArgumentParser
from multiprocessing import Queue, Value
from worker import Worker
from utils.utility import exec_cmd
from utils.utility import progress_bar
from utils import __VERSION__


class  GenerateData(object):
    def __init__(self, config={}):
        self.config = config

        self.max_workers = int(self.config["workers"])
        # Logging
        self.logging_path = "/var/log/dss"
        self.logging_level = "INFO"
        if "logging" in config:
            self.logging_path = config["logging"].get("path", "/var/log/dss")
            self.logging_level = config["logging"].get("level", "INFO")

        if config.get("debug", False):
            self.logging_level = "DEBUG"
        self.logger = None
        self.logger_status = Value('i', 0)  # 0=NOT-STARTED, 1=RUNNING, 2=STOPPED
        self.logger_queue = Queue()
        # Workers
        self.workers = []
        self.workers_finished = Value('i', 0)

        # Queue
        self.task_queue = Queue()
        self.list_queue = Queue()

        # File copy count
        self.file_paths = [] # List of files with complete path
        self.total_listed_file = 0
        self.file_copy_count = Value('i', 0)

        # Defalut variables
        self.s3_config = {}

    def start(self):
        """
        Start the data generation operation.
        It does all of the steps in following sequence.
        :return: None
        """
        # Start Logger
        self.start_logger()
        # Configure setup
        self.configure()
        # Start LISTing of files/objects
        self.indexing()
        # Copy with evenly distributed set through process.
        self.copy()
        # Progress
        self.progress()
        # Summary
        self.summary()

    def stop(self):
        """
        Stop the tool
        :return:
        """
        #Very end
        self.stop_logger()

    def configure(self):
        """
        Configure the switches from the specified configuration file.
        :return:
        """
        try:
            # Data type
            self.source_data_type = self.config["source"]["type"].lower()
            self.destination_data_type = self.config["destination"]["type"].lower()

            # Set source data dirs/prefixes
            self.source_s3_config = None
            if self.source_data_type == "fs":
                self.source_data_dirs = self.config["source"]["storage"]["fs"]["paths"]

            elif self.source_data_type == "s3":
                self.source_data_dirs = self.config["source"]["storage"]["s3"]["prefixes"]
                self.source_s3_config = self.config["source"]["storage"]["s3"]


            # Destination
            self.destination_s3_config = None
            if self.destination_data_type == "fs":
                self.destination_data_dirs = self.config["destination"]["storage"]["fs"]["paths"]
            elif self.destination_data_type == "s3":
                self.destination_data_dirs = self.config["destination"]["storage"]["s3"]["prefixes"]
                self.destination_s3_config = self.config["destination"]["storage"]["s3"]

            self.replication_factor = self.config["replication"]["factor"]
            self.replication_size = self.config["replication"]["max_size"]  # Desired size


        except Exception as e:
            self.logger.fatal(e)
            sys.exit("Configuration failure, Exit!")

    def indexing(self):
        """
        source_data_dirs: [path1,path2,path3,path4,path5]
        mas_workers: Maximum number of works to be used for indexing, say 3
        paths: [[path1,path4],[path2,path5],[path3]]
        :return:
        """
        index = 0
        paths = [] # List of List - A list of set of paths
        list_workers_count = self.max_workers
        if self.max_workers > len(self.source_data_dirs):
            list_workers_count = len(self.source_data_dirs)

        for data_dir in self.source_data_dirs:
            if len(paths) < list_workers_count:
                paths.append([])
            if index >= list_workers_count:
                index = 0  # Reset counter
            paths[index].append(data_dir)
            index += 1
        start_listing_time = time.monotonic()
        s3_client = None
        # Distribute load among the max_workers.
        for worker_id in range(list_workers_count):
            w = Worker(id=worker_id,
                       operation="list",
                       source_storage_type=self.source_data_type,
                       source_s3_config=self.source_s3_config,
                       source_data_dirs=paths[worker_id],
                       list_queue=self.list_queue,
                       worker_finished=self.workers_finished,
                       logger=self.logger
                       )
            w.start()
            self.workers.append(w)
        # Aggregate all files listed by workers
        self.logger.info("Started listing with {} workers".format(list_workers_count))

        while self.workers_finished.value < list_workers_count:
            while self.list_queue.qsize() > 0:
                listed_files = self.list_queue.get()
                listed_files_count = len(listed_files)
                self.total_listed_file += listed_files_count
                self.file_paths.extend(listed_files)
        end_listing_time = time.monotonic()
        if not self.list_queue:
            self.logger.fatal("Couldn't list files, exit application")
            sys.exit()
        self.listing_time = "{:0.4f}".format(end_listing_time - start_listing_time)
        self.logger.info("Total files listed: {}, Time: {} seconds".format(self.total_listed_file, self.listing_time))
        # Clear all workers
        #self.workers = []
        for w in self.workers:
            w.__del__()
        #self.logger.info(self.file_paths)


    def copy(self):
        """
        Split the dataset evenly based on number of workers
        create workers
        wait for all of them to stop
        :return:
        """
        self.logger.info(f"Started Copy Operation for source:{self.source_data_type}"
                         f", destination:{self.destination_data_type}")
        total_files_in_a_set = int(self.total_listed_file / self.max_workers)
        index = 0

        # Default variables
        s3_client =None
        finished_distribution = False
        # Create workers and assign list of files.
        # Distribute load among the max_workers.
        for worker_id in range(self.max_workers):
            max_set_index = index + total_files_in_a_set
            if self.total_listed_file - max_set_index < total_files_in_a_set:
                max_set_index += self.total_listed_file - max_set_index
            # Case where number of files less that max_workers
            if self.total_listed_file < self.max_workers:
                max_set_index = self.total_listed_file
                finished_distribution = True

            self.logger.info(f"Worker-{worker_id} , StartIndex-{index}, EndIndex-{max_set_index}")
            w = Worker(id=worker_id,
                       operation="copy",
                       replication_factor=self.replication_factor,
                       source_storage_type=self.source_data_type,
                       destination_storage_type=self.destination_data_type,
                       file_paths=self.file_paths[index: max_set_index],
                       list_queue=self.list_queue,
                       worker_finished=self.workers_finished,
                       file_copy_count=self.file_copy_count,
                       source_s3_config=self.source_s3_config,
                       destination_s3_config=self.destination_s3_config,
                       destination_dirs=self.destination_data_dirs,
                       logger=self.logger
                       )
            w.start()
            self.workers.append(w)
            index += total_files_in_a_set
            if finished_distribution:
                break


    def progress(self):
        """
        Track the progress of copy operation
        A shared atomic variable is used to track the copy count.
        Each workers increase the atomic variable when copy is done for each file.

        Check the "copy_count" atomic variable and print int the console.
        Check the atomic variable and print into the log when 10%,20% etc is covered.
        :return:
        """
        copy_workers_count = len(self.workers)
        while self.workers_finished.value < copy_workers_count:
            progress_string = f"File Copied:{self.file_copy_count.value}"
            progress_bar(progress_string)
            #time.sleep(0.01)

    def summary(self):
        """
        It writes the summary of file copy operations.
        :return:
        """
        time.sleep(1)
        self.logger.info("\n****************** Summary *******************")
        self.logger.info(f"FileCopy: Source:{self.source_data_type}, Destination:{self.destination_data_type}")
        self.logger.info(f"Total Source Files: {self.total_listed_file}")
        self.logger.info(f"Overall Listing/Indexing Time: {self.listing_time}")
        self.logger.info(f"Replication Factor: {self.replication_factor}")
        self.logger.info(f"Destination dirs/prefixes count: {len(self.destination_data_dirs)}")
        self.logger.info(f"Total Copied Files: {self.file_copy_count.value}")

    def start_logger(self):
        """
        Start Multiprocessing logger
        :return:
        """
        self.logger = MultiprocessingLogger(self.logger_queue,
                                            self.logger_status
                                            )
        self.logger.config(self.logging_path,
                           __file__,
                           self.logging_level)
        self.logger.start()
        self.logger.info("** SyntheticDataGenerator - VERSION:{} **".format(__VERSION__))
        self.logger.info("Started Logger with {} mode!".format(self.logging_level))

    def stop_logger(self):
        """
        Stop multiprocessing logger
        :return:
        """
        self.logger.stop()


if __name__ == "__main__":
    params = ArgumentParser()
    config_obj = Config(params)
    config = config_obj.get_config()
    gd = GenerateData(config)
    gd.start()
    gd.stop()

