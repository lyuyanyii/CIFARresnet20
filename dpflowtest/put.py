from dpflow import control, OutputPipe

q = OutputPipe("lyy-pipe-test", buffer_size = 100)

with control(io = [q]):
	epoc = 0
	while True:
		q.put_pyobj(epoc)
		print(epoc)
		epoc += 1

