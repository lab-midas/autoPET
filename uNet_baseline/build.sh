#!/usr/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t unet_baseline "$SCRIPTPATH"
