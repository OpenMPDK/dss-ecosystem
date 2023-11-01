# Test the .NET Application


## Requirements

- CentOS 7.x
- GCC
- DSS cluster deployed with the DSS client library
- .NET 

## Install .NET


```shell
sudo rpm -Uvh https://packages.microsoft.com/config/centos/7/packages-microsoft-prod.rpm && \
sudo yum install dotnet-sdk-7.0
```


## Compile the .NET project and run the test

```shell
LD_LIBRARY_PATH=/usr/dss/client-library dotnet run "http://msl-ssg-vm03-tcp-0:9000" "minio" "minio123"
```
