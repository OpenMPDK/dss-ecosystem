#!/bin/bash

for i in {1..8}; do
	~/jerry/minio/minio server --address :900${i} /tmp/tenant${i} &
done
