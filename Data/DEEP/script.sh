#!/bin/bash

# URL of the dataset
URL="https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin"

# Use wget to download the dataset
wget "$URL" -O deep1B.fbin
