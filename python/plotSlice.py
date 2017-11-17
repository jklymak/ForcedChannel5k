import numpy
import numpy as np
import scipy
from numpy import squeeze, shape
from numpy import *
from MITgcmutils import rdmds
import sys, getopt, time
import pickle
from os import mkdir
from os import system
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


u0 = 5
runname='CW3dfull01U%02d'%u0
basep='../results/'+runname+'/'+'_Model/input/'
dn = 720
timestd = dn*np.array([42000/dn])
timestd = [27000]
timestd = np.arange(36000,39601,180)
timestd = [36000]
#timestd = [0]

try:
    mkdir(outdir)
except:
    pass

oldfiles=[]

dz = rdmds(basep + 'DRF',machineformat='l');
z = squeeze(rdmds(basep +'RC',machineformat='l')[:,0,0]);
zf = rdmds(basep +'RF',machineformat='l');
#T = rdmds(basep + '0*/T',num,machineformat='l',fill_value=numpy.NaN)
#x = rdmds(basep + '0*/XC',machineformat='l',fill_value=numpy.NaN)
xc = squeeze(rdmds(basep + 'XC',machineformat='l',fill_value=numpy.NaN)[0,:])
x0 = numpy.mean(xc)
xc=xc-x0
inx = np.where(np.abs(xc)<250.e3)[0]
print inx
print('inx %d %d'%(inx[0],inx[-1]))
x=xc
yc = squeeze(rdmds(basep + 'YC',machineformat='l',fill_value=numpy.NaN)[:,0])
y=yc
ny = np.shape(y)[0]
print('Ny=%d'% ny)

nx0=shape(x)[0]
xg = squeeze(rdmds(basep + 'XG',machineformat='l',fill_value=numpy.NaN)[0,:])
xg=xg-x0
dxc = squeeze(rdmds(basep + 'DXC',machineformat='l',fill_value=numpy.NaN)[0,:])
#dxg = squeeze(rdmds(basep + '0*/DXG',machineformat='l',fill_value=numpy.NaN)[0,:])

N=shape(x)[0]
nx = shape(x)[0]
print nx

x = xc/1000.
y=yc/1000.


#hFacC=rdmds(basep + '0*/hFacC',machineformat='l',fill_value=numpy.NaN)
Dep = rdmds(basep+'Depth',machineformat='l')
#for num in range(72000,86800,400):

ind = 5

reg=(inx[0],inx[-1],ny/2,ny/2+1)
print reg

u = 0.*np.array(timestd)

for indd,timeind in enumerate(timestd):
#for timeind in [780]:
    print( timeind)
    num=timeind
    tds = ['T','U']
    D=dict()
    for td in tds:
        D[td] = squeeze(rdmds(basep +td,num,region=reg,machineformat='l',fill_value=numpy.NaN))
	D['Depth']=Dep[0]
	D['x']=x
	D['y']=y
	D['z']=z
	D['dzc']=dz
        if 0:
	    with open(outdir+'Snap%s%05d.pickle'%(td,timeind),'wb') as f:
                pickle.dump(D,f)
    print(D.keys())
    D['U']=np.ma.masked_where(D['T']==0,D['U'])
    D['T']=np.ma.masked_where(D['T']==0,D['T'])
    u[indd] = D['U'][200,200]
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(x[inx],z,D['U']-u0/100.,cmap='RdBu_r',vmin=-0.2,vmax=0.2,rasterized=True)
    print(np.shape(D['T']))
    print(np.shape(x[inx]))
    print(D['T'][:,0])
    ax.contour(x[inx[:-1]],z,D['T'],np.linspace(26.,28.,num=50),colors='0.4',linewidths=0.5)
    fig.colorbar(pcm,ax=ax)
    fig.savefig('../figs/%s%05d.png'%(runname,timeind))

todir = 'valdez.seos.uvic.ca:leewaves15/'
syscom= 'rsync -av ../figs/* '+todir+runname
print syscom
ret=system(syscom)
#print ret
print u

exit()

