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


import argparse
import concurrent.futures
import dss
import hashlib
import itertools
import logging
import logging.config
import os
import random
import sys
import threading
import time
import uuid

from concurrent.futures import ProcessPoolExecutor
from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists

object_data = None
object_data_md5 = None
object_data_size = 0
data_dir = 'dss_client_data/'
data_file_ref = None
data_md5_file_ref = None

log_level = 'INFO'
log_file = 'dss_benchmark.log'

g_end_point = None
g_access_key = None
g_secret_key = None
g_logger = None
g_instance_id = 0
g_endpoints_per_cluster = 256


def _set_logger(log_filename, module):
    log_dict = {
        'version': 1.0,
        'disable_existing_loggers': True,
        'formatters': {
            'simple': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(pathname)s [%(lineno)d] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'level': log_level,
                'formatter': 'simple',
                'filename': log_filename,
                'encoding': 'utf-8'
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['file']
        }
    }

    logging.config.dictConfig(log_dict)
    logger_handle = logging.getLogger(module)
    return logger_handle


# https://julien.danjou.info/atomic-lock-free-counters-in-python/
class FastWriteCounter(object):
    def __init__(self):
        self._number_of_read = 0
        self._counter = itertools.count()
        self._read_lock = threading.Lock()

    def increment(self):
        next(self._counter)

    def value(self):
        with self._read_lock:
            value = next(self._counter) - self._number_of_read
            self._number_of_read += 1
        return value


class FastWriteCounterOld(object):
    def __init__(self):
        self.value = 0
        self._read_lock = threading.Lock()

    def increment(self):
        # value = None
        with self._read_lock:
            value = self.value
            self.value += 1
        return value

    def value(self):
        return self.value


class MinioClient(object):
    def __init__(self, endpoint, access_key, secret_key, logger, region=None):
        self.client = Minio(endpoint, access_key, secret_key, secure=False)
        self.logger = logger
        self.bucket = 'dss0'
        self.logger.info("INFO: Bucket Name - {}".format(self.bucket))
        try:
            self.client.make_bucket(self.bucket)
            self.logger.info("INFO: New bucket created - {}".format(self.bucket))
        except BucketAlreadyOwnedByYou:
            self.logger.info("INFO: Bucket {} already own by you".format(self.bucket))
        except BucketAlreadyExists:
            self.logger.info("WARNING: Bucket {} already exist".format(self.bucket))
        except:
            self.logger.exception('Exception in creating bucket')

    def list_objects(self, prefix=None):
        try:
            objects = self.client.list_objects(bucket_name=self.bucket, prefix=prefix, recursive=True)
        except:
            self.logger.exception("Exception in listObjects")
            raise
        return objects

    def put_object(self, key=None, value=None):
        try:
            self.client.fput_object(bucket_name=self.bucket, object_name=key[1:],
                                    file_path=value, content_type="text/plain")
        except:
            self.logger.exception("Exception in put_object %s", key)
            raise
        return True

    def del_object(self, key=None):
        try:
            """
            objects = self.list(bucket,prefix,recursive)
            #self.client.remove_objects(bucket_name=bucket,objects_iter=objects)
            for object in objects:
              print("Removing object - {}".format(object.object_name))
              self.client.remove_object(bucket_name=bucket,object_name=object.object_name)  # Require object name,
            """
            self.client.remove_object(bucket_name=self.bucket, object_name=key)
        except:
            self.logger.exception("Exception in del_object %s", key)
            raise
        return True

    def get_object(self, key, value):
        """
        Download the object based on the specified object key to the specified file path.
        :param key: Required
        :param value: Required , destination file path
        :return:
        """
        try:
            if value != '/dev/null':
                self.client.fget_object(self.bucket, key, value)
            else:
                response = self.client.get_object(self.bucket, key)
                response.close()
                response.release_conn()
        except:
            self.logger.exception("Exception in get_object %s", key)
            raise

    def get_objects(self, prefix):
        return self.list_objects(prefix)


class DSSClient(object):
    def __init__(self, endpoint, access_key, secret_key, logger, region=None):
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint = endpoint
        self.region = region
        self.logger = logger
        self.buffer = bytearray(object_data_size)
        self.uuid = str(uuid.uuid3(uuid.NAMESPACE_DNS,
                                   str(uuid.getnode()) + str(g_instance_id)))
        try:
            option = dss.clientOption()
            option.maxConnections = 1
            self.client = dss.createClient(self.endpoint, self.access_key, self.secret_key, option,
                                           uuid=self.uuid, endpoints_per_cluster=g_endpoints_per_cluster)
        except dss.NetworkError as e:
            self.logger.error('Exception in instantiating client - Invalid Endpoint or Network error')
            self.client = None
            raise e
        except Exception as e:
            self.logger.error('Exception in instantiating client')
            self.client = None
            raise e

    def put_object(self, key, value):
        try:
            if value:
                self.client.putObject(key, value)
            else:
                self.client.putObjectBuffer(key, object_data, object_data_size)
        except Exception as e:
            self.logger.exception('Failed to put the key %s', key)
            raise e

    def get_object(self, key, value):
        try:
            if value:
                self.client.getObject(key, value)
            else:
                self.client.getObjectBuffer(key, self.buffer)
        except Exception as e:
            self.logger.exception('Failed to get the object %s', key)
            raise e

    def get_objects(self, prefix):
        objects = list()
        try:
            obj_iter = self.client.getObjects(prefix)
            while True:
                try:
                    for file in obj_iter:
                        objects.append(file)
                except dss.NoIterator:
                    break
        except Exception as e:
            self.logger.exception('Failed to get the objects for prefix %s', prefix)
            raise e

        return objects

    def list_objects(self, prefix):
        try:
            list_objs = self.client.listObjects(prefix)
            return list_objs
        except Exception as e:
            self.logger.exception('Failed to list the objects for prefix %s', prefix)
            raise e

    def del_object(self, key):
        try:
            self.client.deleteObject(key)
        except Exception as e:
            self.logger.exception('Failed to get the object %s', key)
            raise e


def run_data_put_prepare(thr_id, key_prefix, num_ios=0, duration=0):
    count = 0
    for count in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = os.path.join(data_dir, key)
        with open(filename, 'wb') as f:
            f.write(object_data)
    return count + 1, 0


def run_data_put(thr_id, key_prefix, num_ios=0, duration=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return 0, 0
    except:
        logger.exception('Error in creating DSS client connection for thread ID %d', thr_id)
        return 0, 0

    start_time = time.time()
    count = 0
    fail_count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        if duration and time.time() - start_time > duration:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        if data_dir not in ['/dev/null', 'None']:
            filename = os.path.join(data_dir, key)
        else:
            filename = None
        count = count + 1
        logger.debug('Thread-%d - Uploading the file %s', thr_id, filename)
        try:
            client_conn.put_object(key, filename)
            logger.debug('Thread-%d - Uploading the file %s DONE', thr_id, filename)
        except:
            logger.error('Thread-%d - Uploading the file %s FAILED', thr_id, filename)
            # fail_count += 1
            print(f'Failed to upload the file {filename} by thread {thr_id}. Exiting')
            return count, (num_ios - count)
    end_time = time.time()
    logger.info('PUT objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('PUT objects - Thr_id %d,  total_io_count %d, failed_io %d,  time %d sec', thr_id, count,
                fail_count, (end_time - start_time))
    return count, fail_count


def run_data_put_cleanup(thr_id, key_prefix, num_ios=0, duration=0):
    count = 0
    for count in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = os.path.join(data_dir, key)
        os.unlink(filename)
    return count + 1, 0


def run_data_get(thr_id, key_prefix, num_ios=0, duration=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return 0, 0
    except:
        logger.exception('Error in creating DSS client connection for thread ID %d', thr_id)
        return 0, 0

    start_time = time.time()
    count = 0
    fail_count = 0
    while True:
        if duration and time.time() - start_time > duration:
            break
        if num_ios:
            if count >= num_ios:
                break
            key_id = random.randint(0, num_ios - 1)
        else:
            key_id = count
        key = '%s-object-%s-%d' % (key_prefix, thr_id, key_id)
        if data_dir == '/dev/null':
            filename = data_dir
        elif data_dir == 'None':
            filename = None
        else:
            filename = os.path.join(data_dir, key)

        count = count + 1
        try:
            client_conn.get_object(key, filename)
        except:
            logger.error('Thread-%d - Downloading the file %s FAILED', thr_id, key)
            print(f'Failed to download the file {key} by thread {thr_id}. Exiting')
            # fail_count += 1
            return count, (num_ios - count)

    end_time = time.time()

    logger.info('GET objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('GET objects - Thr_id %d,  total_io_count %d, failed_io %d,  time %d sec', thr_id, count,
                fail_count, end_time - start_time)
    return count, fail_count


def check_data_after_get(thr_id, key_prefix, num_ios=0, duration=0):
    logger = g_logger
    fail_count = 0
    for i in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, i)
        if data_dir != '/dev/null':
            filename = os.path.join(data_dir, key)
            with open(filename, 'rb') as file:
                output = file.read()
                output_md5 = hashlib.md5(output).hexdigest()
                logger.debug('Obj MD5 - %s, Actual MD5 - %s', output_md5, object_data_md5)
                if output_md5 != object_data_md5:
                    logger.error("Key %s didn't match with object data", key)
                    fail_count += 1
    logger.info('CHECK data - total %d, mismatch %d', num_ios, fail_count)
    return num_ios, fail_count


def run_data_del(thr_id, key_prefix, num_ios=0, duration=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return 0, 0
    except:
        logger.exception('Error in creating DSS client connection for thread ID %d', thr_id)
        return 0, 0

    start_time = time.time()
    count = 0
    fail_count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        if duration and time.time() - start_time > duration:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        count = count + 1
        try:
            client_conn.del_object(key)
        except:
            logger.error('Thread-%d - Deleting the key %s FAILED', thr_id, key)
            print(f'Failed to delete the key {key} by thread {thr_id}. Exiting')
            # fail_count += 1
            return count, (num_ios - count)

    end_time = time.time()

    logger.info('DEL objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('DEL objects - Thr_id %d,  total_io_count %d, failed_io %d,  time %d sec', thr_id, count,
                fail_count, (end_time - start_time))
    return count, fail_count


def run_data_list(thr_id, key_prefix, num_ios=0, duration=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return 0, 0
    except:
        logger.exception('Error in creating DSS client connection')
        return 0, 0

    start_time = time.time()
    count = 0
    fail_count = 0
    key = '%s-object-%s' % (key_prefix, thr_id)
    try:
        objects = client_conn.get_objects(key)
        for file in objects:
            count += 1
    except:
        logger.exception('Failed to list objects with prefix %s', key)
        fail_count += 1

    end_time = time.time()

    logger.info('LIST objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('LIST objects - Thr_id %d,  count %d, failed %d,  time %d sec', thr_id, count,
                fail_count, (end_time - start_time))
    return count, fail_count


def get_cpu_count():
    cpus = os.listdir('/sys/class/cpuid')
    return len(cpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--access-key', dest='access_key', help='Access Key of the Minio server', required=True)
    parser.add_argument('-s', '--secret-key', dest='secret_key', help='Secret Key of the Minio server', required=True)
    parser.add_argument('-u', '--endpoint-url', dest='endpoint_url', help='Endpoint URL of MINIO server', required=True)
    parser.add_argument('-d', '--duration', dest='duration', help='Duration in seconds', default=0)
    parser.add_argument('-l', '--loops', dest='total_loops', help='Number of loops to run (default: 1)', default=1)
    parser.add_argument('-n', '--num_ios', dest='num_ios', help='Number of IOs to do (default: 100)',
                        type=int, default=100, required=True)
    parser.add_argument('-o', '--op_type', dest='op_type',
                        help='Type of IO (1 - PUT, 2 - GET, 3 - DEL, 4 - LIST, '
                             '5 - GET WITH INTEGRITY CHECK, 8 - PREPARE DATA FOR PUT, '
                             '9 - CLEANUP, 0 - PUT/GET/DEL)',
                        type=int, choices=[0, 1, 2, 3, 4, 5, 8, 9], default=0)
    parser.add_argument('-t', '--num_threads', dest='thr_cnt', help='Number of threads to start (default: 1)',
                        type=int, default=1)
    parser.add_argument('-z', '--object-size', dest='object_size', help='size of object in KB (default:1024)',
                        type=int, default=1024)
    parser.add_argument('-p', '--key-prefix', dest='key_prefix', help='Key prefix for the object name', default='dss')
    # parser.add_argument('-c', '--objects_per_thread', dest='objects_per_thread',
    #                     help='number of objects per thread already written')
    parser.add_argument('-x', '--data-dir', dest='data_dir',
                        help='Data directory to read from/write to. '
                             'If /dev/null is given, then the objects are not saved. Only applicable for GET calls '
                             '(default: ./dss_client_data)',
                        default='dss_client_data')
    parser.add_argument('-i', '--id', dest='instance_id', help='Instance ID (default: 0)',
                        type=int, default=0)
    parser.add_argument('-ep', '--endpoints-per-cluster', dest='ep_per_cluster',
                        help='Number of endpoints to pick from each logical cluster (default: 256)',
                        type=int, default=256)
    args = parser.parse_args()

    logger = _set_logger(log_file, 'dss_benchmark')
    logger.info('Input args %s', str(args))

    if args.data_dir:
        data_dir = args.data_dir

    prog_path = os.path.abspath(os.path.join(os.getcwd(), os.path.dirname(sys.argv[0])))
    data_file_ref = os.path.join(prog_path, 'datafile_ref')
    data_md5_file_ref = os.path.join(prog_path, 'datafile_ref.md5')

    g_logger = logger
    g_end_point = args.endpoint_url
    g_access_key = args.access_key
    g_secret_key = args.secret_key
    g_instance_id = args.instance_id
    g_endpoints_per_cluster = args.ep_per_cluster

    cpu_count = get_cpu_count()
    if cpu_count < 8 and args.thr_cnt > cpu_count / 2:
        max_worker_threads = int(cpu_count / 2)
    else:
        max_worker_threads = args.thr_cnt

    if data_dir != '/dev/null':
        l_dir = data_dir
        split_val = args.key_prefix.rsplit('/', 1)
        if len(split_val) > 1:
            l_dir = os.path.join(l_dir, split_val[0])
        if not os.path.exists(l_dir):
            os.makedirs(l_dir)

    if args.op_type == 5 and data_dir == '/dev/null':
        print('Invalid usage. Need a valid directory for integrity check')
        sys.exit(-1)

    object_data_size = args.object_size * 1024
    if args.op_type in [0, 1, 8]:
        object_data = os.urandom(args.object_size * 1024)
        object_data_md5 = hashlib.md5(object_data).hexdigest()
        with open(data_file_ref, 'wb') as f:
            f.write(object_data)
            logger.info('Saved the object data to datafile_ref')
        with open(data_md5_file_ref, 'w') as f:
            f.write(object_data_md5)
            logger.info('Saved the object data md5 hash to datafile_ref')
    elif data_dir != '/dev/null':
        if not os.stat(data_file_ref):
            logger.info('No data file present to check. Exiting')
            sys.exit(-1)

        with open(data_file_ref, 'rb') as f:
            object_data = f.read()
        with open(data_md5_file_ref, 'r') as f:
            object_data_md5 = f.read()

        if object_data_md5 != hashlib.md5(object_data).hexdigest():
            logger.error('Wrong digest info. Exiting')
            sys.exit(-1)

    fn_list = {0: [run_data_put, run_data_get, run_data_del],
               1: [run_data_put],
               2: [run_data_get],
               3: [run_data_del],
               4: [run_data_list],
               5: [run_data_get, check_data_after_get],
               8: [run_data_put_prepare],
               9: [run_data_put_cleanup]
               }

    for loop in range(args.total_loops):
        for fn in fn_list[args.op_type]:
            io_count = 0
            fail_io_count = 0
            with ProcessPoolExecutor(max_workers=max_worker_threads) as executor:
                task_ids = iter(range(args.thr_cnt))
                start_time = time.time()
                try:
                    futures = {
                        executor.submit(fn, task_id, args.key_prefix, args.num_ios, int(args.duration))
                        for task_id in itertools.islice(task_ids, max_worker_threads)
                    }
                    while futures:
                        done, futures = concurrent.futures.wait(
                            futures, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            res = fut.result()
                            io_count += res[0]
                            fail_io_count += res[1]
                            logger.info('Task result - %s', str(res))

                        for task_id in itertools.islice(task_ids, len(done)):
                            futures.add(executor.submit(fn, task_id, args.key_prefix, args.num_ios,
                                                        int(args.duration)))
                except Exception as ex:
                    logger.exception('Exception in running the process')
                    print(f'Exception occurred while running the process- {ex}. Check the log file')
                    sys.exit(-1)

                end_time = time.time()
                time_taken = end_time - start_time
                logger.info('Time taken for fn %s - %d, start time - %d, end time - %d', str(fn),
                            end_time - start_time, start_time, end_time)
                if fn == run_data_put_prepare:
                    print('Data is prepared for PUT calls')
                elif fn == run_data_put_cleanup:
                    print('Prepared data is removed from the local directory')

                if fn in [run_data_put, run_data_get]:
                    actual_ios = io_count - fail_io_count
                    total_io_size = (actual_ios * args.object_size * 1024)
                    throughput = float(total_io_size) / (time_taken * 1024 * 1024 * 1024)
                    logger.info('Throughput - %f GB/s', throughput)
                    if fail_io_count:
                        print('Failed IO - %s' % fail_io_count)
                    if fn == run_data_get:
                        print('GET Throughput - %f GB/s' % throughput)
                    elif fn == run_data_put:
                        print('PUT Throughput - %f GB/s' % throughput)

                if fn == run_data_del:
                    actual_ios = io_count - fail_io_count
                    logger.info('DEL Operations/sec - %f', int(actual_ios) / time_taken)
                    if fail_io_count:
                        print('Failed IO - %s' % fail_io_count)
                    print('DEL Operations per sec - %f' % (int(actual_ios) / time_taken))

                if fn == run_data_list:
                    actual_ios = io_count - fail_io_count
                    logger.info('LIST Operation completed in - %d s, objects found - %d', time_taken, actual_ios)
                    if fail_io_count:
                        print('Failed IO - %s' % fail_io_count)
                    print('LIST Operation completed in - %d s, objects found - %d' % (time_taken, actual_ios))
