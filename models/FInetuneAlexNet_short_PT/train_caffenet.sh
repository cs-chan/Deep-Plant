#!/usr/bin/env sh



./build/tools/caffe train -solver /FInetuneAlexNet_short_PT/solver.prototxt -weights /models/bvlc_alexnet.caffemodel -gpu 0

