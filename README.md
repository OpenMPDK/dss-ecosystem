# dss-ecosystem

hi

DSS Ecosystem software consists of several software packages to interact with and benchmark DSS Object storage.

## DSS Client Library

A customized S3 client library based on [AWS SDK for C++](https://github.com/aws/aws-sdk-cpp) to support multiple logical clusters.
DSS Client Library hides the underlying buckets from the user, allowing access to objects using corresponding keys.

[DSS Client Library README](./dss_client/README.md)

## DSS Datamover

A horizontally scalable distributed command line tool to interact with DSS Object storage, supporting the following operations:

- `PUT`: Uploaded file systems data to DSS object storage
  - Upload files from several client nodes and aggregate the upload status.
- `LIST`: List the object keys from DSS object storage
  - Supports parallel listing of object keys from a single node
  - Supports distributed listing of object keys and aggregation of keys.
- `GET`: Download objects from DSS object storage
  - Download objects to a shared file space
- `DELETE`: Remove objects from DSS object storage

Additionally, there is support for `RESUME` operation to upload additional files excluded from a previous `PUT`.

[DSS Datamover README](./dss_datamover/README.md)

## Benchmark Tools

The following are custom benchmark tools developed to measure performance of DSS Object storage.

### DSS AI-Benchmark

A benchmark tool developed based on machine leaning (ML) framework such as PyTorch and TensorFlow to evaluate
the read access performance of the DSS Object storage through ML model training.


[DSS AI_benhcmark README](./dss_ai_benchmark/README.md)

### DSS s3-benchmark

A customized benchmark tool based on the original [Wasabi s3-benchmark](https://github.com/wasabi-tech/s3-benchmark) tool.

[DSS s3-benchmark README](./dss_s3benchmark/README.md)
