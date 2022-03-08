#!/usr/bin/env bash
# shellcheck disable=SC1090
# The Clear BSD License
#
# Copyright (c) 2022 Samsung Electronics Co., Ltd.
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
DSS_CLIENT_DIR=$(realpath "$SCRIPT_DIR/..")
DSS_ECOSYSTEM_DIR=$(realpath "$DSS_CLIENT_DIR/..")
BUILD_DIR="$DSS_CLIENT_DIR/build"
STAGING_DIR="$BUILD_DIR/staging"
# ARTIFACTS_DIR="$ANSIBLE_DIR/artifacts"

# Remove Client Library build dir and artifacts if they exist
rm -rf "$BUILD_DIR"
rm -f "$DSS_CLIENT_DIR"/*.tgz

# Load GCC
. "$SCRIPT_DIR/load_gcc.sh"

# Check for libaws libs
if [ ! -f /usr/local/lib64/libaws-c-common.so ]
then
    die "Missing AWS libs. Build using GCC 5.1.0: https://codeload.github.com/aws/aws-sdk-cpp/tar.gz/refs/tags/1.8.99#/aws-sdk-cpp-1.8.99.tar.gz"
fi

# Build Client Library
mkdir -p "$DSS_CLIENT_DIR/build"
pushd "$DSS_CLIENT_DIR/build"
    CXX=g++ cmake3 ../
    make -j
popd

# Get dss-ecosystem release string
pushd "$DSS_ECOSYSTEM_DIR"
    # Get release string
    git fetch --tags
    RELEASESTRING=$(git describe --tags --exact-match || git rev-parse --short HEAD)
popd

# Create dss_client staging dir
mkdir -p "$STAGING_DIR"

# Stage release and create release tarball
cp "$BUILD_DIR"/*.so "$STAGING_DIR"

# Copy Client benchmark directory to staging directory
cp -r "$DSS_CLIENT_DIR/benchmark" "$STAGING_DIR"

# Create Client Library release tarball
pushd "$STAGING_DIR"
    tar czfv "dss_client-$RELEASESTRING.tgz" ./*
    mv "dss_client-$RELEASESTRING.tgz" "$DSS_CLIENT_DIR"
popd

# Remove staging directory
rm -rf "$STAGING_DIR"
