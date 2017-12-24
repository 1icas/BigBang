#ifndef LAYER_TYPE_MACRO_H
#define LAYER_TYPE_MACRO_H

#include "../base.h"

#define ACTIVATION_FUNC_TYPE "activation_func"
#define COST_FUNC_TYPE "cost_func"
#define NEURON_FUNC_TYPE "neuron_func"
#define IMAGE_FUNC_TYPE "image_func"
//image
#define IMAGE_TYPE "Image"
#define DATA_LAYER "Data"
//cost func
#define MSE_LAYER_TYPE "MSE"
#define SOFTMAX_COST_LAYER_TYPE "SoftmaxCost"
#define LOG_LIKELIHOOD "LogLikelihood"
//activation func
#define SIGMOID_LAYER_TYPE "Sigmoid"
#define RELU_LAYER_TYPE "Relu"
#define SOFTMAX_LAYER_TYPE "Softmax"
#define DROPOUT_LAYER_TYPE "Dropout"
//neuron func
#define CONV_LAYER_TYPE "Conv"
#define INNER_PRODUCT_LAYER_TYPE "InnerProduct"
#define POOLING_LAYER_TYPE "Pooling"




#endif
