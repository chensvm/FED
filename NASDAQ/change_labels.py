import sys

inputfilename = sys.argv[1]
outputfilename = sys.argv[2]

prev_price = 0.0

with open(outputfilename,'w') as fout:
	with open(inputfilename,'r') as fin:
		for ind,line in enumerate(fin):
			if ind == 0:
				fout.write(line)
			elif ind == 1:
				s = line.split(',')
				prev_price = float(s[1])
			else:
				s = line.split(',')
				label = '-1'
				if float(s[1]) >= prev_price:
					label = '1'
				else:
					label = '0'
				prev_price = float(s[1])
				fout.write(stock + s[0] + ',' + label + '\n')

