#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

#docker build --no-cache -t autopet_baseline "$SCRIPTPATH"
docker build -t autopet_baseline "$SCRIPTPATH"
docker build -f "$SCRIPTPATH/Dockerfile.eval" -t autopet_eval .