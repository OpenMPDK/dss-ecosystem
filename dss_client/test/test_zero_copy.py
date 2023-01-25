import dss
import hashlib
import numpy as np
import os
import random

FILE_COUNT = 20

# from memory_profiler import profile


# @profile
def check_get_with_zero_copy_with_numpy(filename, print_enable=False):
    xs = np.asarray(bytearray(1024 * 1024))
    c = dss.createClient('202.0.0.1:9000', 'minio', 'minio123')
    content_length = c.getObjectNumpyBuffer(filename, xs)
    if print_enable:
        buffer = xs.tobytes().decode('utf-8').rstrip('\x00')
        print("*" * 32)
        print("Function check_zero_copy_with_numpy")
        print(f"Buffer [{buffer}]")
        print(f"Content Length {content_length}")
        print("*" * 32)
    return xs.tobytes(), content_length


def check_get_with_zero_copy(filename, print_enable=False):
    xs = bytearray(1024 * 1024)
    c = dss.createClient('202.0.0.1:9000', 'minio', 'minio123')
    content_length = c.getObjectBuffer(filename, xs)
    if print_enable:
        buffer = xs.decode('utf-8').rstrip('\x00')
        print("Function check_zero_copy")
        print("*" * 32)
        print(f"Buffer [{buffer}]")
        print(f"Content Length {content_length}")
        print("*" * 32)
    return xs, content_length


def check_put_buffer():
    filename = 'MyDummy'
    xs = bytearray(1024 * 1024)
    c = dss.createClient('202.0.0.1:9000', 'minio', 'minio123')
    size = random.randint(128 * 1024, 1024 * 1024)
    xs[:size] = os.urandom(size)
    object_data = xs[:size]
    c.putObjectBuffer(filename, xs, size)
    data_in_md5 = hashlib.md5(object_data).hexdigest()

    ys = bytearray(1024 * 1024)
    length = c.getObjectBuffer(filename, ys)
    mv = memoryview(ys)
    data_out = mv[:length]
    data_out_md5 = hashlib.md5(data_out).hexdigest()
    assert (data_in_md5 == data_out_md5)
    assert (size == length)


def integrity_check():
    invalid_files = False
    c = dss.createClient('202.0.0.1:9000', 'minio', 'minio123')
    for i in range(FILE_COUNT):
        filename = "up_file_" + str(i)
        size = random.randint(128 * 1024, 1024 * 1024)
        object_data = os.urandom(size)
        with open(filename, 'wb') as f:
            f.write(object_data)
            print(f"Created {filename} with size {size}")
        c.putObject(filename, filename)

    for fn in [check_get_with_zero_copy, check_get_with_zero_copy_with_numpy]:
        print("*" * 32)
        print(f"Validating with {str(fn)}")
        print("*" * 32)
        for i in range(FILE_COUNT):
            key = "up_file_" + str(i)
            filename = "down_file_" + str(i)
            with open(key, 'rb') as file:
                data_in = file.read()
                data_in_md5 = hashlib.md5(data_in).hexdigest()
            data_out, length = fn(key)
            mv = memoryview(data_out)
            data_to_write = mv[:length]
            data_out_md5 = hashlib.md5(data_to_write).hexdigest()
            if data_in_md5 != data_out_md5:
                print(f"File {key} is not in sync with {filename}")
                invalid_files = True
                with open(filename, 'wb') as f:
                    f.write(data_to_write)
            else:
                print(f"Object {key} is valid")

    if not invalid_files:
        for i in range(FILE_COUNT):
            filename = "up_file_" + str(i)
            os.unlink(filename)


if __name__ == '__main__':
    # check_zero_copy_with_numpy('test.py', True)
    # check_zero_copy('test.py', True)
    integrity_check()
