#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=./model_organ/solver.prototxt \
--weights=./two-path_CNN/multi_path_vgg_imagenet_iter_426094.caffemodel \
--gpu=0


