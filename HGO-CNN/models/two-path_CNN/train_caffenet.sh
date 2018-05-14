#!/usr/bin/env sh

caffe train \
    --solver=/home/exx/Documents/two-path_CNN/solver.prototxt \
--weights=/data/PlantClef/VGG_ILSVRC_16_layers.caffemodel \
--gpu=0


