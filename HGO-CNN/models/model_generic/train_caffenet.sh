#!/usr/bin/env sh

./build/tools/caffe train -solver ./model_generic/solver.prototxt -weights ./model_organ/PlantClef_vgg_organ_iter_89229.caffemodel -gpu 0

