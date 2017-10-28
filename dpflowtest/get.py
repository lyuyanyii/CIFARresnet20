from dpflow import InputPipe, control

p = InputPipe("lyy-pipe-test", buffer_size = 100)

with control(io = [p]):
	for i in range(100):
		print(p.get())
