#!/usr/bin/env bash

# SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="4g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create autopet_baseline-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --gpus=all \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        autopet_baseline

docker run --rm \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/results.json | python -m json.tool

docker run --rm \
        -v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/input/:/input/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm autopet_baseline-output-$VOLUME_SUFFIX
