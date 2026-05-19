#!/bin/bash

hostname=$(cat /etc/hostname)

METADATA_FILENAME_PATH=$1/metadata_$hostname.txt

[[ -d $METADATA_FILENAME_PATH ]] && rm $METADATA_FILENAME_PATH

python3 --version >> $METADATA_FILENAME_PATH
conda --version >> $METADATA_FILENAME_PATH
pip --version >> $METADATA_FILENAME_PATH

echo "=========" >> $METADATA_FILENAME_PATH
echo "CPUINFO" >> $METADATA_FILENAME_PATH
echo "=========" >> $METADATA_FILENAME_PATH

cat /proc/cpuinfo >> $METADATA_FILENAME_PATH

echo "=========" >> $METADATA_FILENAME_PATH
echo "GPU INFO" >> $METADATA_FILENAME_PATH
echo "=========" >> $METADATA_FILENAME_PATH

nvidia-smi -q -i 0 >> $METADATA_FILENAME_PATH
cat /usr/local/cuda/version.* >> $METADATA_FILENAME_PATH

echo "=========" >> $METADATA_FILENAME_PATH
echo "MEMINFO" >> $METADATA_FILENAME_PATH
echo "=========" >> $METADATA_FILENAME_PATH

cat /proc/meminfo >> $METADATA_FILENAME_PATH

echo "=========" >> $METADATA_FILENAME_PATH
echo "OS INFO" >> $METADATA_FILENAME_PATH
echo "=========" >> $METADATA_FILENAME_PATH

cat /etc/os-release >> $METADATA_FILENAME_PATH
cat /proc/version >> $METADATA_FILENAME_PATH
