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
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DSS_ELBENCHO_DIR=$(realpath "$SCRIPT_DIR/../")
DSS_ECOSYSTEM_DIR=$(realpath "$SCRIPT_DIR/../..")
TOPDIR=$(realpath "$DSS_ECOSYSTEM_DIR/..")
DSS_CLIENT_DIR="$DSS_ECOSYSTEM_DIR/dss_client"
DSS_CLIENT_LIB_DIR="$DSS_CLIENT_DIR/build"
DSS_CLIENT_INC_DIR="$DSS_CLIENT_DIR/include"
NKV_SDK_DIR="$TOPDIR/dss-sdk/host_out"
NKV_LIB_DIR="$NKV_SDK_DIR/lib"
STAGING_DIR="$DSS_ELBENCHO_DIR/staging"

# Print a message to console and return non-zero
die()
{
    echo "$*"
    exit 1
}

# Check for libaws libs
if [ ! -f /usr/local/lib64/libaws-c-common.so ]
then
    die "Missing AWS libs. Build using devtoolset-11: https://github.com/breuner/aws-sdk-cpp.git"
fi

if [ ! -f "$NKV_LIB_DIR/librdd_cl.so" ]; then
        die "librdd_cl.so is missing. Please clone and build dss-sdk: https://github.com/openMPDK/dss-sdk or download the nkv-sdk artifact and change the path accordingly"
fi

echo "Cleaning old artifacts"
rm -f "$DSS_ELBENCHO_DIR/dss_elbencho-*.tgz"

echo "Building the elbencho binary"
make clean
source /opt/rh/devtoolset-11/enable 
make S3_SUPPORT=1 AWS_INCLUDE_DIR=/usr/local/include/ AWS_LIB_DIR=/usr/local/lib64  DSS_LIB_DIR="$DSS_CLIENT_LIB_DIR" NKV_LIB_DIR="$NKV_LIB_DIR" DSS_INCLUDE_DIR="$DSS_CLIENT_INC_DIR" BUILD-VERBOSE=1 -j 2

echo "Packaging the binaries to dss_elbencho-<release>.tgz"
# Get dss-ecosystem release string
pushd "$DSS_ECOSYSTEM_DIR"
    # Get release string
    git fetch --tags
    RELEASESTRING=$(git describe --tags --exact-match || git rev-parse --short HEAD)
popd

mkdir -p "$STAGING_DIR"
cp "$DSS_ELBENCHO_DIR/bin/elbencho" "$STAGING_DIR"

pushd "$STAGING_DIR"
	tar zvcf "$DSS_ELBENCHO_DIR/dss_elbencho-$RELEASESTRING.tgz" ./*
popd

rm -rf "$STAGING_DIR"
