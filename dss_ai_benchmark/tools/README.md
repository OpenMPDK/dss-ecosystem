# dss_dnn_benchmark

--------------------
To build the script
--------------------

go build create_imgs.go

---------
To Run it
---------

rm -rf <dir-path>/*;ulimit -n 1024000; ./create_imgs -p <dir-path> -f 10000 -maxFilesPerThread 500 -size 10000 -seed 123 -rootPrefix si04 -startingFile ./sample.png -maxSize 1048576 2>&1 | tee result.txt
