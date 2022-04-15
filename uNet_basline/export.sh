#!/usr/bin/env bash

./build.sh

docker save unet_basline | gzip -c > uNet_basline.tar.gz
