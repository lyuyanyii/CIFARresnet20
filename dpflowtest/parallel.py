import os
import argparse
from multiprocessing import Process as P

def worker(name):
	os.system("mdl " + name)


parser = argparse.ArgumentParser()
parser.add_argument("t", type = int)
parser.add_argument("m")
args = parser.parse_args()

lis = []
for i in range(args.t):
	proc = P(target = worker, args = (args.m, ))
	proc.start()
	lis.append(proc)

for i in lis:
	i.join()
