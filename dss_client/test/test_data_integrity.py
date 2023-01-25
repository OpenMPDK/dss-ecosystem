import argparse
import dss
import hashlib
import os

data_hash_map = dict()


def command_line_parser():
    parser = argparse.ArgumentParser(description='Distributed version of S3 CLI to perform Data integrity')
    parser.add_argument("--count", "-c", type=int, default=1, required=False, action='store',
                        help='Number of files (Default: 1)')
    parser.add_argument("--debug", "-d", required=False, action='store_true',
                        help='Debug flag (Default: False)')
    parser.add_argument("--endpoint", "-e", type=str, required=True, action='store',
                        help='Cluster endpoint IP address (Default: <minio_ip_addr>:<port>)')
    parser.add_argument("--file_name_prefix", "-f", type=str, default="testfile", required=False, action='store',
                        help='File name prefix(Default: testfile)')
    parser.add_argument("--path_prefix", "-p", type=str, default="/samsung/s3test/", required=False, action='store',
                        help='Prefix path for the files to be uploaded. Creates dir if not present'
                             '(Default: /samsung/s3test/)')
    parser.add_argument("--password", "-pw", type=str, default="minio123", required=False, action='store',
                        help='Minio endpoint password (Default: minio123)')
    parser.add_argument("--size", "-s", type=int, default=1, required=False, action='store',
                        help='Size in MB (Default: 1)')
    parser.add_argument("--user", "-u", type=str, default="minio", required=False, action='store',
                        help='Minio endpoint admin (Default: minio)')
    return parser


def create_client_connection(endpoint, user='minio', pw='minio123'):
    c = dss.createClient(endpoint, user, pw)
    return c


def get_object(client_handle, filename, size, print_enable=False):
    buffer = bytearray(size)
    content_length = client_handle.getObjectBuffer(filename, buffer)
    mv = memoryview(buffer)
    data_out = mv[:content_length]
    data_out_md5 = hashlib.md5(data_out).hexdigest()
    if print_enable:
        print(f"GetObject: {filename} length: {content_length} md5sum: {data_out_md5}")
    return data_out_md5


def put_object(client_handle, filename, size, print_enable=False):
    buffer = os.urandom(size)
    client_handle.putObjectBuffer(filename, buffer, size)
    data_in_md5 = hashlib.md5(buffer).hexdigest()
    if print_enable:
        print(f"PutObject: {filename} length: {size} md5sum: {data_in_md5}")
    return data_in_md5


def data_integrity_check(endpoint, user, password, prefix, file_name_prefix, size, file_count, debug=False):
    try:
        c = create_client_connection(endpoint, user, password)
        print(f"Uploading objects with the name {prefix + file_name_prefix}")
        print("-" * 64)
        os.makedirs(prefix, 0o755, True)
        full_filename_prefix = '/'.join([prefix, file_name_prefix])
        for i in range(file_count):
            filename = full_filename_prefix + str(i)
            if debug:
                print(f"Uploading file {filename}")
            object_data = os.urandom(size)
            with open(filename, 'wb') as f:
                f.write(object_data)
            data_in_md5 = put_object(c, filename, size, debug)
            data_hash_map[filename] = data_in_md5
        print("-" * 64)

        print(f"Downloading objects with the name {full_filename_prefix} and verifying")
        print("-" * 64)
        for i in range(file_count):
            filename = full_filename_prefix + str(i)
            if debug:
                print(f"Downloading file {filename}")
            data_out_md5 = get_object(c, filename, size, debug)
            if data_out_md5 != data_hash_map[filename]:
                print(f"File {filename} contents mismatched")
        print("-" * 64)

        print(f"Cleaning up objects with the name {full_filename_prefix} on local disk")
        print("-" * 64)
        for i in range(file_count):
            filename = full_filename_prefix + str(i)
            os.unlink(filename)
        print("-" * 64)

    except Exception as e:
        print(f"Exception {e}")


if __name__ == '__main__':
    p = command_line_parser()
    args = p.parse_args()
    data_integrity_check(args.endpoint, args.user, args.password, args.path_prefix, args.file_name_prefix, args.size, args.count, args.debug)

