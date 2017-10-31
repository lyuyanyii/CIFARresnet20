import numpy as np

from megskull.network import Network
from megskull.opr.all import (
    Conv2D, Pooling2D, FullyConnected, Softmax,
	    CrossEntropyLoss, Dropout
		)
from megskull.opr.helper.elemwise_trans import ReLU, Identity
from megskull.graph.query import GroupNode
from megskull.opr.netsrc import DataProvider
import megskull.opr.helper.param_init as pinit
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.regularizer import BatchNormalization as BN
import megskull.opr.arith as arith

global idx
idx = 0

def conv_bn(inp, ker_shape, stride, padding, out_chl, isrelu):
	global idx
	idx += 1
	l1 = Conv2D(
		"conv{}".format(idx), inp, kernel_shape = ker_shape, stride = stride, padding = padding,
		output_nr_channel = out_chl,
		W = G(mean = 0, std = (2 / (ker_shape**2 * inp.partial_shape[1]))**0.5),
		nonlinearity = {True:ReLU(), False:Identity()}[isrelu]
		)
	l2 = BN("bn{}".format(idx), l1)
	return l2

def res_layer(inp, chl):
	pre = inp
	inp = conv_bn(inp, 3, 1, 1, chl, True)
	inp = conv_bn(inp, 3, 1, 1, chl, False)
	inp = arith.ReLU(inp + pre)
	return inp

def res_block(inp, chl, n):
	if chl == 16:
		inp = res_layer(inp, chl)
	else:
		pre = inp
		inp = conv_bn(inp, 3, 2, 1, chl, True)
		inp = conv_bn(inp, 3, 1, 1, chl, False)
		inp = inp + conv_bn(pre, 1, 2, 0, chl, False)
		inp = arith.ReLU(inp)
	
	for i in range(n - 1):
		inp = res_layer(inp, chl)
	
	return inp

def make_network(minibatch_size = 128):
	patch_size = 32
	inp = DataProvider("data", shape = (minibatch_size, 3, patch_size, patch_size))
	label = DataProvider("label", shape = (minibatch_size, ))

	lay = conv_bn(inp, 3, 1, 1, 16, True)

	n = 3
	lis = [16, 32, 64]
	for i in lis:
		lay = res_block(lay, i, n)
	
	#global average pooling
	feature = lay.mean(axis = 2).mean(axis = 2)
	pred = Softmax("pred", FullyConnected(
		"fc0", feature, output_dim = 10,
		W = G(mean = 0, std = (2 / 64)**0.5),
		b = C(0),
		nonlinearity = Identity()
		))
	
	network = Network(outputs = [pred])
	network.loss_var = CrossEntropyLoss(pred, label)
	return network

