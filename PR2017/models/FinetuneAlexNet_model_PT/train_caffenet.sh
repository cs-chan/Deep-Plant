#!/usr/bin/env sh


./build/tools/caffe train -solver /FinetuneAlexNet model_PT/solver.prototxt -weights /models/bvlc_alexnet.caffemodel -gpu 0





