import os
import argparse
from multiprocessing import Process as P
from multiprocessing import Queue, Lock
from dpflow import control, OutputPipe
import numpy as np

def augment(img):
	padding1 = np.zeros((3, 32, 4))
	img = np.concatenate([padding1, img, padding1], axis = 2)
	padding2 = np.zeros((3, 4, 40))
	img = np.concatenate([padding2, img, padding2], axis = 1)
	x, y = np.random.randint(8, size = 2)
	img = img[:, x:x + 32, y : y + 32]
	return img

import msgpack
import msgpack_numpy as m

def worker(data, pname, que, lock, is_train):
	p = OutputPipe(pname, buffer_size = 200)
	lock.acquire()
	with control(io = [p]):
		lock.release()
		while True:
			idx = que.get()
			que.put((int(idx) + 1) % len(data[0]))
			img = np.array(data[0][int(idx)])
			img = np.resize(img.astype(np.float32), (3, 32, 32))
			img = (img - 128) / 256
			if is_train:
				img = augment(img)
			#a = msgpack.packb([img, data[1][int(idx)]], default = m.encode)
			#b = msgpack.unpackb(a, object_hook = m.decode)
			#print(np.array(b[0]).shape, b[1])
			p.put_pyobj([np.array(img), int(data[1][int(idx)])])
			print("put {} data {} successfully".format({True:"train", False:"valid"}[is_train], int(idx)))

def load_data(name):
	import pickle
	with open(name, "rb") as fo:
		dic = pickle.load(fo, encoding = 'bytes')
	return dic


parser = argparse.ArgumentParser()
parser.add_argument("t", type = int)
#parser.add_argument("m")
args = parser.parse_args()

if True:
	dics = []
	for i in range(1, 6):
		dics.append(load_data("/home/liuyanyi02/CIFAR/cifar-10-batches-py/data_batch_{}".format(i)))

	data = np.array(dics[0][b'data'])
	labels = np.array(dics[0][b'labels'])
	for i in range(1, 5):
		data = np.concatenate([data, np.array(dics[i][b'data'])], axis = 0)
		labels = np.concatenate([labels, np.array(dics[i][b'labels'])], axis = 0)
	train_dataset = [data[:45000], labels[:45000]]
	valid_dataset = [data[45000:], labels[45000:]]
	lis = []
	que = Queue(1)
	que.put(0)
	lock = Lock()

	p = "lyy.CIFAR10.resnet20.train"
	for i in range(args.t):
		proc = P(target = worker, args = (train_dataset, p, que, lock, True))
		proc.start()
		lis.append(proc)
	
	que_val = Queue(1)
	que_val.put(0)
	p_val = "lyy.CIFAR10.resnet20.valid"
	proc = P(target = worker, args = (valid_dataset, p_val, que_val, lock, False))
	proc.start()
	lis.append(proc)
	
	for i in lis:
		i.join()

else:
	test_dic = load_data("~/CIFAR/cifar-10-batches-py/test_batch")
