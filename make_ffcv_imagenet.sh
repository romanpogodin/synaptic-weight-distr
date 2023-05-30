#!/bin/bash

# change according tp your setup; more RAM is better (e.g. 200G); the final dataset will be roughly
# double the size of ImageNet (about 300G).
conda activate your_venv

# The script assumes the current directory has ./imagenet/train and ./imagenet/val folder,
# and ./make_ffcv_imagenet.py

WRITE_DIR="./imagenet"
IMAGENET_DIR="./imagenet"
WORKERS=64  # change if needed

write_dataset () {
    write_path="${WRITE_DIR}/${1}${5}_${2}_${3}_${4}.ffcv"
    echo "Writing ImageNet ${1} dataset to ${write_path}"
    python ./make_ffcv_imagenet.py \
        --cfg.dataset=imagenet \
        --cfg.split=${5} \
        --cfg.data_dir=$IMAGENET_DIR/${1} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=${2} \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=${3} \
        --cfg.jpeg_quality=${4} \
        --cfg.num_workers=$WORKERS
}

# Serialize images with (from ffcv imagenet example)
# 500 - 500px side length maximum
# 0.50 - 50% JPEG encoded, 90% raw pixel values
# 90 - quality=90 JPEGs

write_dataset train 500 0.50 90 '_no10k'
write_dataset train 500 0.50 90 '_10k'
write_dataset val 500 0.50 90
