#!/usr/bin/env bash

./build.sh

docker save unet_baseline | gzip -c > uNet_baseline.tar.gz
