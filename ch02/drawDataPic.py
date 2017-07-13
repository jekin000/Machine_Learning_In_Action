import numpy as np
import kNN
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def draw():
	fig = plt.figure()
	ax = fig.add_subplot(111)


	mt,l = kNN.file2matrix('datingTestSet2.txt')

	ax.scatter(mt[:,0],mt[:,1],15.0*np.array(l),15.0*np.array(l))
	fig.savefig('0_1.png')
	return



