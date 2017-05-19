#!/usr/bin/env sh



./build/tools/caffe train -solver /FusionCascade/solver.prototxt -weights /models/bvlc_alexnet.caffemodel -gpu 0
