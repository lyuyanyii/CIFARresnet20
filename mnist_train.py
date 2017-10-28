from meghair.train.env import Environment as Env
env = Env()
from megskull.optimizer import NaiveSGD, OptimizableFunc
from megskull.graph import Function
from megskull.optimizer.momentum import Momentum
import network
from megskull.graph import FpropEnv
from meghair.utils.io import dump

func = OptimizableFunc.make_from_loss_var(network.loss)

Momentum(learning_rate = 1e-2, momentum = 0.9)(func)

func.compile(network.loss)

#env = FpropEnv()
#pred_mgb = env.get_mgbvar(network.pred)
#func_test = env.comp_graph.compile_outonly(pred_mgb)
func_test = Function().compile(network.pred)

import pickle
import gzip
import numpy as np

train_set, valid_set, test_set = pickle.load(gzip.open("mnist.pkl.gz", "rb"), encoding = "latin1")

minibatch_size = network.minibatch_size

l = len(train_set[0])
epoch = 0
for i in range(100000):
	j = i % (l // minibatch_size)
	minibatch = train_set[0][j * minibatch_size : (j + 1) * minibatch_size]
	label = train_set[1][j * minibatch_size : (j + 1) * minibatch_size]
	minibatch = np.array(minibatch).reshape(-1, 1, 28, 28)
	loss = func(input_mat = minibatch, label = label)
	if j == 0:
		print("*****")
		print("epoch = ", epoch)
		epoch += 1
		print("loss = ", loss)
		res = func_test(input_mat = np.array(valid_set[0]).reshape(-1, 1, 28, 28))
		res = np.argmax(np.array(res), axis = 1)
		acc = (np.array(res) == np.array(valid_set[1])).mean()
		print("acc = ", acc)
		dump(network.network, open("1.data", "wb"))

