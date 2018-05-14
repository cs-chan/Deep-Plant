#!/usr/bin/env sh


./build/tools/caffe train -solver ./model_species/solver.prototxt -weights ./model_generic/PlantClef_vgg_species__iter_200000.caffemodel -gpu 0



