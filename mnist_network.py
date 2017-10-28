from megskull.opr.all import DataProvider, Conv2D, Pooling2D, Exp, Log, Softmax, CrossEntropyLoss
from megskull.opr.all import FullyConnected as FC
from megskull.opr.helper.param_init import ConstantParamInitializer as C
from megskull.opr.helper.param_init import AutoGaussianParamInitializer as G
from megskull.opr.helper.elemwise_trans import Identity, ReLU
from megskull.network import Network
import numpy as np

minibatch_size = 20
img_size = 28

input_mat = DataProvider(name = "input_mat", 
			shape = (minibatch_size, 1, img_size, img_size))
conv1 = Conv2D("conv1", input_mat, kernel_shape = 3, output_nr_channel = 5, 
			W = G(mean = 0.0001, std = (1 / (3 * 3))**0.5),
			b = C(0),
			padding = (1, 1),
			nonlinearity = ReLU())
conv2 = Conv2D("conv2", conv1, kernel_shape = 3, output_nr_channel = 5,
			W = G(mean = 0.0001, std = (1 / (5 * 3 * 3))**0.5),
			b = C(0),
			padding = (1, 1),
			nonlinearity = ReLU())
pooling1 = Pooling2D("pooling1", conv2, window = (2, 2), mode = "max")

conv3 = Conv2D("conv3", pooling1, kernel_shape = 3, output_nr_channel = 10, 
			W = G(mean = 0.0001, std = (1 / (5 * 3 * 3))**0.5),
			b = C(0),
			padding = (1, 1),
			nonlinearity = ReLU())
conv4 = Conv2D("conv4", conv3, kernel_shape = 3, output_nr_channel = 10,
			W = G(mean = 0.0001, std = (1 / (10 * 3 * 3))**0.5),
			b = C(0),
			padding = (1, 1),
			nonlinearity = ReLU())
pooling2 = Pooling2D("pooling2", conv4, window = (2, 2), mode = "max")

feature = pooling2.reshape((-1, 7 * 7 * 10))
fc1 = FC("fc1", feature, output_dim = 100,
			W = G(mean = 0.0001, std = (1 / 490)**0.5),
			b = C(0),
			nonlinearity = ReLU())
fc2 = FC("fc2", fc1, output_dim = 10,
			W = G(mean = 0, std = (1 / 100)**0.5),
			b = C(0),
			nonlinearity = Identity())
#output_mat = Exp(fc2) / Exp(fc2).sum(axis = 1).dimshuffle(0, 'x')
pred = Softmax("pred", fc2)

label = DataProvider(name = "label", shape = (minibatch_size, ), dtype = np.int32)
#loss = -Log(indexing_one_hot(output_mat, 1, label)).mean()
loss = CrossEntropyLoss(pred, label)

network = Network(pred, loss)
