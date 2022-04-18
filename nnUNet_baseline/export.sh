#!/usr/bin/bash

./build.sh

docker save autopet_baseline | gzip -c > autoPET_baseline.tar.gz
