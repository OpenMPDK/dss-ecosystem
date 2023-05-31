#!/usr/bin/env bash
# shellcheck disable=SC1090,SC1091
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

set -e

# Set path variables
SCRIPT_DIR=$(readlink -f "$(dirname "$0")")
DSS_S3BENCHMARK_DIR=$(realpath "$SCRIPT_DIR/..")
DSS_ECOSYSTEM_DIR=$(realpath "$DSS_S3BENCHMARK_DIR/..")
DSS_SDK_DIR="$DSS_ECOSYSTEM_DIR/../dss-sdk/host_out"
LIB_DIR="-L$DSS_SDK_DIR/lib -L$DSS_ECOSYSTEM_DIR/dss_client/build"
INCLUDE_DIR="-I$DSS_SDK_DIR/include -I$DSS_ECOSYSTEM_DIR/dss_client/include"

# Print a message to console and return non-zero
die()
{
    echo "$*"
    exit 1
}

if [ ! -d "${DSS_SDK_DIR}" -o ! -f "${DSS_SDK_DIR}/include/rdd_cl.h" -o ! -f "${DSS_SDK_DIR}/lib/librdd_cl.so" ]; then
    die "dss-sdk repo is missing. Download the repo github.com/OpenMPDK/dss-sdk and compile"
fi

if [ ! -f "${DSS_ECOSYSTEM_DIR}/dss_client/build/libdss.so" ]; then
    die "dss_client library is missing"
fi

# Install golang RPMs
#rpm -q golang || yum install -y golang

export CGO_CFLAGS="-std=gnu99 $INCLUDE_DIR"
export CGO_LDFLAGS="$LIB_DIR -lrdmacm -libverbs -ldss -lrdd_cl"
export GO111MODULE=off
export GODIR="$DSS_ECOSYSTEM_DIR/go-repos"
export GOPATH="$GODIR"
export PATH="$PATH:$GODIR/bin"
export GOCACHE="$GODIR/cache"

echo 'Downloading go repos'
if [ -d "$GODIR" ]; then
    rm -rf "$GODIR"
    rm -rf "$GOCACHE"
fi
mkdir -p "$GODIR"
mkdir -p "$GOCACHE"
go get -u github.com/aws/aws-sdk-go/aws/...
go get -u github.com/aws/aws-sdk-go/service/...
go get -u code.cloudfoundry.org/bytefmt

echo 'Building S3 benchmark'
go build -o s3-benchmark s3-benchmark.go 

echo 'cleaning cache and repos'
rm -rf "$GODIR"
rm -rf "$GOCACHE"
