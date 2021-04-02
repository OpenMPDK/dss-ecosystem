import argparse
import dss
import hashlib
import itertools
import logging
import logging.config
import os
import sys
import threading
import time

from concurrent.futures import ProcessPoolExecutor, as_completed

object_data = None
object_data_md5 = None
data_dir = 'dss_client_data/'
data_file_ref = None
data_md5_file_ref = None

log_level = 'INFO'
log_file = 'dss_benchmark.log'

g_end_point = None
g_access_key = None
g_secret_key = None
g_logger = None


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


class FastWriteCounter_old(object):
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


class DSSClient(object):
    def __init__(self, endpoint, access_key, secret_key, logger, region=None):
        try:
            self.access_key = access_key
            self.secret_key = secret_key
            self.endpoint = endpoint
            self.region = region
            self.logger = logger
            option = dss.clientOption()
            option.maxConnections = 1
            try:
                self.client = dss.createClient(self.endpoint, self.access_key, self.secret_key, option)
            except:
                self.logger.exception('Error in creating client connection')
        except Exception as e:
            self.logger.exception(e)
            raise e

    def put_object(self, key, value):
        try:
            self.client.putObject(key, value)
        except Exception as e:
            self.logger.exception('Failed to put the key %s', key)
            raise e

    def get_object(self, key, value):
        try:
            self.client.getObject(key, value)
        except Exception as e:
            self.logger.exception('Failed to get the object %s', key)
            raise e

    def get_objects(self, prefix):
        try:
            self.client.getObjects(prefix, len(prefix))
        except Exception as e:
            self.logger.exception('Failed to list the objects for prefix %s', prefix)
            raise e

    def list_objects(self, prefix):
        try:
            self.client.listObjects(prefix)
        except Exception as e:
            self.logger.exception('Failed to list the objects for prefix %s', prefix)
            raise e

    def del_object(self, key):
        try:
            self.client.deleteObject(key)
        except Exception as e:
            self.logger.exception('Failed to get the object %s', key)
            raise e


def run_data_put_prepare(thr_id, key_prefix, num_ios=0):
    for count in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = os.path.join(data_dir, key)
        with open(filename, 'wb') as f:
            f.write(object_data)


def run_data_put(thr_id, key_prefix, num_ios=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return
    except:
        logger.error('Error in creating DSS client connection')
        return

    start_time = time.time()
    count = 0
    fail_count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = os.path.join(data_dir, key)
        count = count + 1
        logger.debug('Thread-%d - Uploading the file %s', thr_id, filename)
        try:
            client_conn.put_object(key, filename)
            logger.debug('Thread-%d - Uploading the file %s DONE', thr_id, filename)
        except:
            logger.debug('Thread-%d - Uploading the file %s FAILED', thr_id, filename)
            fail_count += 1
    end_time = time.time()
    logger.info('PUT objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('PUT objects - Thr_id %d,  count %d, failed %d,  time %d sec', thr_id, count,
                fail_count, (end_time - start_time))


def run_data_put_cleanup(thr_id, key_prefix, num_ios=0):
    for count in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = os.path.join(data_dir, key)
        os.unlink(filename)


def run_data_get(thr_id, key_prefix, num_ios=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return
    except:
        logger.error('Error in creating DSS client connection')
        return

    start_time = time.time()
    count = 0
    fail_count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = os.path.join(data_dir, key)
        count = count + 1
        try:
            client_conn.get_object(key, filename)
        except:
            fail_count += 1

    end_time = time.time()

    logger.info('GET objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('GET objects - Thr_id %d,  count %d, failed %d,  time %d sec', thr_id, count,
                fail_count, end_time - start_time)


def check_data_after_get(thr_id, key_prefix, num_ios=0):
    logger = g_logger
    fail_count = 0
    for i in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, i)
        filename = os.path.join(data_dir, key)
        with open(filename, 'rb') as file:
            output = file.read()
            output_md5 = hashlib.md5(output).hexdigest()
            logger.debug('Obj MD5 - %s, Actual MD5 - %s', output_md5, object_data_md5)
            if output_md5 != object_data_md5:
                logger.error("Key %s didn't match with object data", key)
                fail_count +=1
    logger.info('CHECK data - total %d, mismatch %d', num_ios, fail_count)


def run_data_del(thr_id, key_prefix, num_ios=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return
    except:
        logger.error('Error in creating DSS client connection')
        return

    start_time = time.time()
    count = 0
    fail_count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        count = count + 1
        try:
            client_conn.del_object(key)
        except:
            fail_count += 1

    end_time = time.time()

    logger.info('DEL objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('DEL objects - Thr_id %d,  count %d, failed %d,  time %d us', thr_id, count,
                fail_count, (end_time - start_time) * 10**6)


def run_data_list(thr_id, key_prefix, num_ios=0):
    logger = g_logger
    try:
        client_conn = DSSClient(g_end_point, g_access_key, g_secret_key, logger)
        if not client_conn:
            logger.error('Error in creating DSS client connection')
            return
    except:
        logger.error('Error in creating DSS client connection')
        return

    start_time = time.time()
    count = 0
    fail_count = 0
    key = '%s-object-%s' % (key_prefix, thr_id)
    try:
        obj = client_conn.get_objects(key)
        count = len(obj)
    except:
        logger.info('Failed to list objects with prefix %s', key)
        fail_count += 1

    end_time = time.time()

    logger.info('LIST objects - Thr_id %d, Start time - %d, End time - %d', thr_id, start_time, end_time)
    logger.info('LIST objects - Thr_id %d,  count %d, failed %d,  time %d us', thr_id, count,
                fail_count, (end_time - start_time) * 10**6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--access-key', dest='access_key', help='Access Key of the Minio server', required=True)
    parser.add_argument('-s', '--secret-key', dest='secret_key', help='Secret Key of the Minio server', required=True)
    parser.add_argument('-u', '--endpoint-url', dest='endpoint_url', help='Endpoint URL of MINIO server', required=True)
    # parser.add_argument('-d', '--duration', dest='duration', help='Duration in seconds (default: 10)', default=10)
    parser.add_argument('-l', '--loops', dest='total_loops', help='Number of loops to run (default: 1)', default=1)
    parser.add_argument('-n', '--num_ios', dest='num_ios', help='Number of IOs to do (default: 1)',
                        type=int, default=1, required=True)
    parser.add_argument('-o', '--op_type', dest='op_type',
                        help='Type of IO (1 - PUT, 2 - GET, 3 - DEL, 4 - LIST, '
                             '8 - PREPARE DATA FOR PUT, '
                             '9 - CLEANUP, 0 - PUT/GET/DEL)',
                        type=int, choices=[0, 1, 2, 3, 4, 8, 9], default=0)
    parser.add_argument('-t', '--num_threads', dest='thr_cnt', help='Number of threads to start (default: 1)',
                        type=int, default=1)
    parser.add_argument('-z', '--object-size', dest='object_size', help='size of object in KB (default:1024)',
                        type=int, default=1024)
    parser.add_argument('-p', '--key-prefix', dest='key_prefix', help='Key prefix for the object name', default='dss')
    # parser.add_argument('-c', '--objects_per_thread', dest='objects_per_thread',
    #                     help='number of objects per thread already written')
    parser.add_argument('-x', '--data-dir', dest='data_dir', help='Data directory to read from/write to',
                        default='dss_client_data')
    args = parser.parse_args()

    logger = _set_logger(log_file, 'dss_benchmark')
    logger.info('Input args %s', str(args))

    if args.data_dir:
        data_dir = args.data_dir

    data_file_ref = os.path.join(data_dir, 'datafile_ref')
    data_md5_file_ref = os.path.join(data_dir, 'data_md5_file_ref')

    g_logger = logger
    g_end_point = args.endpoint_url
    g_access_key = args.access_key
    g_secret_key = args.secret_key

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.op_type in [0, 1, 8]:
        object_data = os.urandom(args.object_size * 1024)
        object_data_md5 = hashlib.md5(object_data).hexdigest()
        with open(data_file_ref, 'wb') as f:
            f.write(object_data)
            logger.info('Saved the object data to datafile_ref')
        with open(data_md5_file_ref, 'w') as f:
            f.write(object_data_md5)
            logger.info('Saved the object data md5 hash to datafile_ref')
    else:
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

    fn_list = {0: [run_data_put_prepare, run_data_put, run_data_get, run_data_del, run_data_put_cleanup],
               1: [run_data_put],
               2: [run_data_get],
               3: [run_data_del],
               # 4: [run_data_list],
               8: [run_data_put_prepare],
               9: [run_data_put_cleanup]
               }

    for loop in range(args.total_loops):
        for fn in fn_list[args.op_type]:
            with ProcessPoolExecutor(max_workers=args.thr_cnt) as executor:
                future_tasks = []
                start_time = time.time()
                for i in range(args.thr_cnt):
                    try:
                        task = executor.submit(fn, i, args.key_prefix,
                                               args.num_ios)
                        future_tasks.append(task)
                    except Exception as e:
                        logger.exception('Error in starting the task')

                for task in as_completed(future_tasks):
                    res = task.result()
                    # logger.info('Task result - %s', str(res))
                end_time = time.time()
                time_taken = end_time - start_time
                logger.info('Time taken for fn %s - %d, start time - %d, end time - %d', str(fn),
                            end_time - start_time, start_time, end_time)
                if fn == run_data_put_prepare:
                    print('Data is prepared for PUT/GET/DEL calls')
                elif fn == run_data_put_cleanup:
                    print('Data is removed from the directory')
                if fn in [run_data_put, run_data_get, run_data_del]:
                    total_io_size = (args.num_ios * args.thr_cnt * args.object_size * 1024)
                    throughput = float(total_io_size)/(time_taken * 1024 * 1024 * 1024)
                    logger.info('Throughput - %f GB/s', throughput)
                    if fn == run_data_get:
                        print('GET Throughput - %f GB/s' % throughput)
                    elif fn == run_data_put:
                        print('PUT Throughput - %f GB/s' % throughput)
                    elif fn == run_data_del:
                        print('DEL Throughput - %f GB/s' % throughput)
