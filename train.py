import argparse
from meghair.train.env import TrainingEnv, Action
from megskull.opr.loss import WeightDecay

from megskull.graph import FpropEnv
import megskull
from dpflow import InputPipe, control
import time

from network import make_network
import numpy as np

import msgpack
import msgpack_numpy as m

minibatch_size = 128
patch_size = 32

def get_minibatch(p, size):
	data = []
	labels = []
	for i in range(size):
		#a = p.get()
		#(img, label) = msgpack.unpackb(a, object_hook = m.decode)
		(img, label) = p.get()
		data.append(img)
		labels.append(label)
	return {"data": np.array(data).astype(np.float32), "label":np.array(labels)}

with TrainingEnv(name = "lyy.resnet20.test", part_count = 2) as env:
	net = make_network(minibatch_size = minibatch_size)
	preloss = net.loss_var
	net.loss_var = WeightDecay(net.loss_var, {"*conv*:W": 1e-4, "*fc*:W": 1e-4})

	train_func = env.make_func_from_loss_var(net.loss_var, "train", train_state = True)
	valid_func = env.make_func_from_loss_var(net.loss_var, "val", train_state = False)

	lr = 0.1
	optimizer = megskull.optimizer.Momentum(lr, 0.9)
	#optimizer.learning_rate = 0.01
	optimizer(train_func)
	
	train_func.comp_graph.share_device_memory_with(valid_func.comp_graph)

	dic = {
		"loss": net.loss_var,
		"pre_loss": preloss,
		"outputs": net.outputs[0]
	}
	train_func.compile(dic)
	valid_func.compile(dic)
	
	env.register_checkpoint_component("network", net)
	env.register_checkpoint_component("opt_state", train_func.optimizer_state)

	tr_p = InputPipe("lyy.CIFAR10.resnet20.train", buffer_size = 1000)
	va_p = InputPipe("lyy.CIFAR10.resnet20.valid", buffer_size = 1000)
	epoch = 0
	EPOCH_NUM = 45000 // 128
	i = 0
	max_acc = 0

	with control(io = [tr_p]):
		with control(io = [va_p]):
	
			while i <= 64000:
				i += 1
				data = get_minibatch(tr_p, minibatch_size)
				out = train_func(data = data['data'], label = data["label"])
				loss = out["pre_loss"]
				print("minibatch = {}, loss = {}".format(i, loss))
				#Learning Rate Adjusting
				if i == 32000 or i == 48000:
					optimizer.learning_rate /= 10
				if i == 64000:
					optimizer.learning_rate = 1e-5
				if i % (EPOCH_NUM) == 0:
					env.save_checkpoint("resnet20.data")
					epoch += 1
					data_val = get_minibatch(va_p, 5000)
					out_val = valid_func(data = data["data"], label = data["label"])
					pred = np.argmax(np.array(out["outputs"]), axis = 1)
					acc = (np.array(pred) == np.array(data["label"])).mean()
					if acc > max_acc and i > 64000:
						max_acc = acc
						env.save_checkpoint("resnet20.data.bestmodel")
					print("epoch = {}, acc = {}, max_acc = {}".format(epoch, acc, max_acc))
					print("**************************")
		
