#!/usr/bin/env bash

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create autopet-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/input/:/input/ \
        -v autopet-output-$VOLUME_SUFFIX:/output/ \
        autopet

docker run --rm \
        -v autopet-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/results.json | python -m json.tool

docker run --rm \
        -v autopet-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/input/:/input/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm autopet-output-$VOLUME_SUFFIX
