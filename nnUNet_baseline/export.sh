#!/usr/bin/env bash

./build.sh

docker save autopet | gzip -c > autoPET.tar.gz
