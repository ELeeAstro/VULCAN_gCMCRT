import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

data = np.loadtxt('zero_alb_J.txt')

p1 = data[:,0]
J1 = data[:,1]/(4.0*np.pi)

data = np.loadtxt('regular_J.txt')

p2 = data[:,0]
J2 = data[:,1]/(4.0*np.pi)

data = np.loadtxt('haze_J.txt')

p3 = data[:,0]
J3 = data[:,1]/(4.0*np.pi)

data = np.loadtxt('isotropic_J.txt')

p6 = data[:,0]
J6 = data[:,1]/(4.0*np.pi)

data = np.loadtxt('normal.txt')

p4 = data[:,0]
J4 = data[:,1]/(4.0*np.pi)

data = np.loadtxt('dir_beam.txt')

p5 = data[:,0]
J5 = data[:,1]/(4.0*np.pi)

fig = plt.figure()

col = sns.color_palette('colorblind')


plt.plot(J1,p1,label='zero albedo',c=col[0],lw=2)
plt.plot(J6,p6,label='isotropic',c=col[1])
plt.plot(J2,p2,label='Rayleigh',c=col[2])
plt.plot(J3,p3,label='haze layer',c=col[3])

plt.plot(J4,p4,label='two-stream',c=col[4],ls='dashed')
plt.plot(J5,p5,label='direct beam',ls='dashed',c='black')

plt.legend(loc='upper left')

plt.yscale('log')
#plt.xscale('log')

plt.xlim(1e6,5.0e6)
plt.ylim(1e-8,100)

ticks = [1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
ticklab = [r'100',r'10',r'1',r'0.1',r'0.01',r'10$^{-3}$',r'10$^{-4}$',r'10$^{-5}$',r'10$^{-6}$',r'10$^{-7}$',r'10$^{-8}$']
plt.yticks(ticks=ticks,labels=ticklab)

plt.xlabel(r'total mean intensity [erg s$^{-1}$ cm$^{-2}$]',fontsize=16)
plt.ylabel(r'pressure [bar]',fontsize=16)

plt.gca().invert_yaxis()

plt.tick_params(axis='both',which='major',labelsize=14)

plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
plt.savefig('MCRT_J_mean.pdf',dpi=144,bbox_inches='tight')


plt.show()