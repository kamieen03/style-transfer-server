#!/usr/bin/env bash
dir=$(dirname $0)

./$dir/compile_full_model.py

models=(vgg_c vgg_s matrix decoder)

for model in ${models[*]}
do
    python3 -m onnxsim $dir/../models/onnx/$model.onnx $dir/../models/onnx/_$model.onnx
    rm $dir/../models/onnx/$model.onnx
    mv $dir/../models/onnx/_$model.onnx $dir/../models/onnx/$model.onnx
    onnx2trt $dir/../models/onnx/$model.onnx -o $dir/../models/trt/$model.trt -v -w 1073741824 -b 1
done

