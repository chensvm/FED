import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys

input_file = sys.argv[1]

loss = []
acc = []
with open(input_file,'r') as fin:
	for line in fin:
		data = line.split(',')
		if len(data) == 3:
			loss.append(data[1].split(':')[1])
			acc.append(data[2].split(':')[1])

plt.plot(loss,'r',acc,'g')
plt.ylim([0,1])

plt.savefig('loss.png')
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()