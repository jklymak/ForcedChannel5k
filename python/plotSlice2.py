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


runname='coarse3dfull01U10'
outdir='/scr/jklymak/scratch/'+runname+'/'
basep='../results/'+runname+'/'+'_Model/input/'
timestd = range(0,36020,1800)
print timestd
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

KEs = np.zeros(len(timestd))

for nn,timeind in enumerate(timestd):
#for timeind in [780]:

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
	with open(outdir+'Snap%s%05d.pickle'%(td,timeind),'wb') as f:
            pickle.dump(D,f)
    print(D.keys())
    D['U']=np.ma.masked_where(D['T']==0,D['U'])
    D['T']=np.ma.masked_where(D['T']==0,D['T'])
    inKE  = np.where(np.diff(x)<1e10)[0]
    print inKE
    areaKE = np.sum((0.*D['U']+1.)*100.*10.)
    print(areaKE)
    print('Hie')
    KE = np.sum(D['U']**2*100.*10.)/areaKE
    print KE
    KEs[nn]=KE
    if 0:
        fig,ax = plt.subplots()
        pcm=ax.pcolormesh(x[inx],z,D['U']-0.1,cmap='RdBu_r',vmin=-0.2,vmax=0.2,rasterized=True)
        ax.contour(x[inx[:-1]],z,D['T'],np.linspace(26.,28.,num=50),colors='0.4',linewidths=0.5)
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Z [m]')
        
        ax.set_title('Time = %d h; KE = %1.5e'%(timeind*5/3600,KE))
        fig.colorbar(pcm,ax=ax)
    
        fig.savefig('../figs/%s%05d.png'%(runname,timeind))

fig,ax = plt.subplots()
ax.plot(np.arange(0,len(KEs)),KEs)
fig.savefig('../figs/%sKEts.png'%(runname))

    
todir = 'valdez.seos.uvic.ca:leewaves15/figs/'
syscom= 'rsync -av ../figs/*.png ' +todir
print syscom
ret=system(syscom)
print ret


exit()

