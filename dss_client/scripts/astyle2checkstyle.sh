#!/usr/bin/env bash
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
TOP_DIR=$(realpath "$SCRIPT_DIR/..")
ASTYLE_CONF="$SCRIPT_DIR/.astylerc"
ASTYLE_EXCLUDES="$SCRIPT_DIR/astyle_excludes.txt"

# Silent pushd and popd
pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

# Run astyle in check-mode
pushd "$TOP_DIR"
    ASTYLE_RESULTS=$(git ls-files '*.[ch]' '*.cpp' '*.cc' '*.cxx' '*.hh' '*.hpp' | grep -vf "$ASTYLE_EXCLUDES" | xargs astyle --options="$ASTYLE_CONF" --dry-run)

    # Generate checkstyle format report
    echo '<?xml version="1.0" encoding="ISO-8859-15"?>'
    echo '<checkstyle version="6.5">'
    echo "$ASTYLE_RESULTS" | while read -r line
    do
        FILENAME=$(echo "$line" | grep -oP "Formatted +\K.+")
        echo "  <file name=\"$FILENAME\">"
        echo "    <error severity=\"error\" message=\"astyle error\" />"
        echo "  </file>"
    done
    echo "</checkstyle>"
popd ''
