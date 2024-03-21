import numpy as np
import matplotlib.pylab as plt

fname = 'He_scat.txt'

data = np.loadtxt(fname)

wl = data[:,0]
xsec = data[:,1]

fig = plt.figure()

plt.plot(wl,xsec)

plt.xscale('log')
plt.yscale('log')


plt.show()
