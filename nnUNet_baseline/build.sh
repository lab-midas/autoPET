#!/usr/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t autopet_baseline "$SCRIPTPATH"
