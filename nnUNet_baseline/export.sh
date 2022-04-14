#!/usr/bin/env bash

./build.sh

docker save autopet_baseline | gzip -c > autoPET_baseline.tar.gz
