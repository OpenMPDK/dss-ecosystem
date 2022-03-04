# dss-ecosystem
The ecosystem software consists couple of software to interact with or benchmark DSS Object storage.
The following section provides a very high level overview of the DSS ecosystem  

## DSS Client library
A customize S3 client library developed based on AWS C++ sdk to support multiple clusters.
The library hide bucket from user and simply allow user to access objects using corresponding keys.
 
## Datamover
A horizontally scalable distributed version of command line tool to interact with DSS Object storage. It supports following operations.
- PUT: Uploaded file systems data to DSS object storage
	- Upload files from several client nodes and aggregate the upload status.
- LIST: List the object keys from DSS object storage
	- Supports parallel listing of object keys from a single node
	- Supports distributed listing of object keys and aggregation of keys.
- GET: Download the objects from DSS object storage 
	- Download objects to a shared file space
- DELETE: Remove objects from DSS object storage

In addition to that, we support RESUME operation to upload files those were not uploaded in the previous run.

## Benchmark Tools
Following are the custom benchmark tools developed to measure performance of DSS Object storage.
### DSS PyTorch 
A benchmark tool developed to showcase the read access performance of the DSS Object storage.
### DSS S3Bench
A customized version of S3Benchmark tool developed on top of original Wasabi S3 Benchmark tool.




