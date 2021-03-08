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

from concurrent.futures import ThreadPoolExecutor, as_completed

object_data = None
object_data_md5 = None
data_dir = '/tmp/'
data_file_ref = data_dir + 'datafile_ref'
data_md5_file_ref = data_dir + 'data_md5_file_ref'

put_count = None
put_fail_count = None
get_count = None
get_fail_count = None
del_count = None
del_fail_count = None

log_level = 'INFO'
log_file = 'dss_benchmark.log'

# Puts per thread
puts_per_thread = {}


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
        value = None
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
            logger.exception('Failed to put the key %s', key)
            raise e

    def get_object(self, key, value):
        try:
            self.client.getObject(key, value)
        except Exception as e:
            logger.exception('Failed to get the object %s', key)
            raise e

    def del_object(self, key):
        try:
            self.client.deleteObject(key)
        except Exception as e:
            logger.exception('Failed to get the object %s', key)
            raise e


def run_data_put(client_conn, thr_id, key_prefix, num_ios=0):
    start_time = time.time()
    count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = data_file_ref
        try:
            client_conn.put_object(key, filename)
            count = count + 1
            put_count.increment()
        except:
            put_fail_count.increment()
    end_time = time.time()

    logger.info('PUT objects - total %d, failed %d, time %d ms', count, put_fail_count.value(),
                (end_time - start_time) * 10**6)
    puts_per_thread[thr_id] = count

    '''
    if num_ios:
        i = 0
        start_time = time.time()
        while i < num_ios:
            key = '%s-object-%s-%d'.format(key_prefix, thr_id, i)
            try:
                client_conn.put_object(key, object_data)
                i = i + 1
                put_count.increment()
            except:
                put_fail_count.increment()
                pass
        end_time = time.time()
        logger.info('PUT objects - count %d - time %d', num_ios, (end_time - start_time))
    else:
        start_time = time.time()
        count = 0
        while True:
            if time.time() - start_time > duration:
                break
            key = '%s-object-%d'.format(key_prefix, count)
            try:
                client_conn.put_object(key, object_data)
                count = count + 1
                put_count.increment()
            except:
                put_fail_count.increment()
                pass
        end_time = time.time()
        logger.info('PUT objects - count %d - time %d', count, (end_time - start_time))
    '''


def run_data_get(client_conn, thr_id, key_prefix, num_ios=0):
    start_time = time.time()
    count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        filename = data_dir + key
        count = count + 1
        try:
            client_conn.get_object(key, filename)
            get_count.increment()
        except:
            get_fail_count.increment()

    end_time = time.time()

    logger.info('GET objects - count %d, failed %d,  time %d ms', count,
                get_fail_count.value(), (end_time - start_time) * 10**6)


def check_data_after_get(client_conn, thr_id, key_prefix, num_ios=0):
    fail_count = FastWriteCounter()
    for i in range(num_ios):
        key = '%s-object-%s-%d' % (key_prefix, thr_id, i)
        filename = data_dir + key
        with open(filename, 'rb') as file:
            output = file.read()
            output_md5 = hashlib.md5(output).hexdigest()
            logger.debug('Obj MD5 - %s, Actual MD5 - %s', output_md5, object_data_md5)
            if output_md5 != object_data_md5:
                logger.error("Key %s didn't match with object data", key)
                fail_count.increment()
    logger.info('CHECK data - total %d, mismatch %d', num_ios, fail_count.value())


def run_data_del(client_conn, thr_id, key_prefix, num_ios=0):
    start_time = time.time()
    count = 0
    while True:
        if num_ios and count >= num_ios:
            break
        key = '%s-object-%s-%d' % (key_prefix, thr_id, count)
        count = count + 1
        try:
            client_conn.del_object(key)
            del_count.increment()
        except:
            del_fail_count.increment()

    end_time = time.time()

    logger.info('DEL objects - count %d, failed %d, time %d ms', count,
                del_fail_count.value(), (end_time - start_time) * 10**6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--access-key', dest='access_key', help='Access Key of the Minio server', required=True)
    parser.add_argument('-s', '--secret-key', dest='secret_key', help='Secret Key of the Minio server', required=True)
    parser.add_argument('-u', '--endpoint-url', dest='endpoint_url', help='Endpoint URL of MINIO server', required=True)
    parser.add_argument('-d', '--duration', dest='duration', help='Duration in seconds (default: 10)', default=10)
    parser.add_argument('-l', '--loops', dest='total_loops', help='Number of loops to run (default: 1)', default=1)
    parser.add_argument('-n', '--num_ios', dest='num_ios', help='Number of IOs to do (default: 1)',
                        type=int, default=1, required=True)
    parser.add_argument('-o', '--op_type', dest='op_type',
                        help='Type of IO (1 - PUT, 2 - GET, 3 - DEL, 4 - LIST, 0 - PUT/GET/DEL)',
                        type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('-t', '--num_threads', dest='thr_cnt', help='Number of threads to start (default: 1)',
                        type=int, default=1)
    parser.add_argument('-z', '--object_size', dest='object_size', help='size of object in KB (default:1024)',
                        type=int, default=1024)
    parser.add_argument('-p', '--key-prefix', dest='key_prefix', help='Key prefix for the object name', default='dss')
    parser.add_argument('-c', '--objects_per_thread', dest='objects_per_thread',
                        help='number of objects per thread already written')
    args = parser.parse_args()

    logger = _set_logger(log_file, 'dss_benchmark')
    logger.info('Input args %s', str(args))

    if args.op_type in [0, 1]:
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

    put_count = FastWriteCounter()
    put_fail_count = FastWriteCounter()
    get_count = FastWriteCounter()
    get_fail_count = FastWriteCounter()
    del_count = FastWriteCounter()
    del_fail_count = FastWriteCounter()

    fn_list = {0: [run_data_put, run_data_get, check_data_after_get, run_data_del],
               1: [run_data_put],
               2: [run_data_get, check_data_after_get],
               3: [run_data_del]
               }

    for loop in range(args.total_loops):
        for fn in fn_list[args.op_type]:
            with ThreadPoolExecutor() as executor:
                future_tasks = []
                for i in range(args.thr_cnt):
                    client_hdl = DSSClient(args.endpoint_url, args.access_key, args.secret_key, logger)
                    if not client_hdl.client:
                        sys.exit(-1)
                    try:
                        task = executor.submit(fn, client_hdl, i, args.key_prefix,
                                               args.num_ios)
                        future_tasks.append(task)
                    except Exception as e:
                        logger.exception('Error in starting the task')

                for task in as_completed(future_tasks):
                    res = task.result()
                    logger.info('Task result - %s', str(res))





