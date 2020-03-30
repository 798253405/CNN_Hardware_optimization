# CNN_Hardware_optimization

for minist.py

This code can run in this enviroment with accuracy about 100%: python:3.7 tensorflow:2.1.
ref: https://github.com/chenweicai/tensorflow-study/blob/master/tf_cnn_mnist.py (python2.x, tensorflow1.x)
If meet problems about input data, can be found here: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials

For AlexNet:
Size of ImageNet2012 dateset is 146 Gb. In this code, only the net was established without reading real pictures and training.
ref: https://github.com/yqtaowhu/MachineLearning/blob/master/deep-learning/AlexNet/alexNet.py

For conv2d_rewrite:
Conv2d rewrited in numpy.
Compared with tf.nn.conv2d, there is an extremly small error of 1e-18.
That might is due to the different precisions used in numpy and tensorflow. 
It will be figured out later.
