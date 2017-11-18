from numpy import *
import numpy as np
#from scipy import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from pylab import *
from shutil import copy
from os import mkdir
import shutil,os,glob
import scipy.signal as scisig
from maketopo import getTopo2D
import logging

logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger(__name__)

runnum = 1
amp = 155.
K0 = 1.8e-4/2./np.pi 
runtype = 'low'  # 'full','filt','low'
setupname=''
u0 = 10
runname='LW16core1km%sU%dAmp%dK%d'%(runtype,u0, amp, K0 * 2 *np.pi * 1e5)
comments = 'Boo'

# to change U we need to edit external_forcing recompile

outdir0='../results/'+runname+'/'

indir =outdir0+'indata/'

## Params for below as per Nikurashin and Ferrari 2010b
H = 4000.
U0 = u0/100.
N0 = 1.e-3
f0 = 1.e-4

# need some info.  This comes from `leewave3d/EmbededRuns.ipynb` on `valdez.seos.uvic.ca`
# the maxx and maxy are for finer scale runs.
dx0=100.
dy0=100.
maxx = 409600.0
maxy = 118400.0


# topography parameters:
useFiltTop=False
useLowTopo=False
gentopo=False # generate the topography.
if runtype=='full':
    gentopo=True
if runtype=='filt':
    useFiltTop=True
elif runtype=='low':
    useLowTopo=True


# model size
nx = 8*52
ny = 2*64
nz = 400

_log.info('nx %d ny %d', nx, ny)

def lininc(n,Dx,dx0):
    a=(Dx-n*dx0)*2./n/(n+1)
    dx = dx0+arange(1.,n+1.,1.)*a
    return dx


#### Set up the output directory
backupmodel=1
if backupmodel:
  try:
    mkdir(outdir0)
  except:
    import datetime
    import time
    ts = time.time()
    st=datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    shutil.move(outdir0[:-1],outdir0[:-1]+'.bak'+st)
    mkdir(outdir0)

    _log.info(outdir0+' Exists')

  outdir=outdir0
  try:
    mkdir(outdir)
  except:
    _log.info(outdir+' Exists')
  outdir=outdir+'input/'
  try:
    mkdir(outdir)
  except:
    _log.info(outdir+' Exists')
  try:
      mkdir(outdir+'/figs/')
  except:
    pass

  copy('gendata.py',outdir)
else:
  outdir=outdir+'input/'

## Copy some other files
_log.info( "Copying files")

try:
  shutil.rmtree(outdir+'/../code/')
except:
  _log.info("code is not there anyhow")
shutil.copytree('../code', outdir+'/../code/')
shutil.copytree('../python', outdir+'/../python/')

try:
  shutil.rmtree(outdir+'/../build/')
except:
  _log.info("build is not there anyhow")
_log.info(outdir+'/../build/')
mkdir(outdir+'/../build/')

# copy any data that is in the local indata
shutil.copytree('../indata/', outdir+'/../indata/')

shutil.copy('../build/mitgcmuvU%02d'%u0, outdir+'/../build/mitgcmuv')
shutil.copy('../build/mitgcmuvU%02d'%u0, outdir+'/../build/mitgcmuv%02d'%u0)
shutil.copy('../build/Makefile', outdir+'/../build/Makefile')
shutil.copy('data', outdir+'/data')
shutil.copy('eedata', outdir)
shutil.copy('data.kl10', outdir)
try:
  shutil.copy('data.kpp', outdir)
except:
  pass
#shutil.copy('data.rbcs', outdir)
try:
    shutil.copy('data.obcs', outdir)
except:
    pass
try:
  shutil.copy('data.diagnostics', outdir)
except:
  pass
try:
  shutil.copy('data.pkg', outdir+'/data.pkg')
except:
  pass
try:
  shutil.copy('data.rbcs', outdir+'/data.rbcs')
except:
  pass

_log.info("Done copying files")

####### Make the grids #########

# Make grids:

##### Dx ######

dx = zeros(nx)+maxx/nx

# dx = zeros(nx)+100.
x=np.cumsum(dx)
x=x-x[0]
maxx=np.max(x)
_log.info('XCoffset=%1.4f'%x[0])

##### Dy ######

dy = zeros(ny)+maxy/ny

# dx = zeros(nx)+100.
y=np.cumsum(dy)
y=y-y[0]
maxy=np.max(y)
_log.info('YCoffset=%1.4f'%y[0])

_log.info('dx %f dy %f', dx[0], dy[0])


# save dx and dy
with open(indir+"/delX.bin", "wb") as f:
  dx.tofile(f)
f.close()
with open(indir+"/delY.bin", "wb") as f:
  dy.tofile(f)
f.close()
# some plots
fig, ax = plt.subplots(2,1)
ax[0].plot(x/1000.,dx)
ax[1].plot(y/1000.,dy)
#xlim([-50,50])
fig.savefig(outdir+'/figs/dx.pdf')

######## Bathy ############
# get the topo:
d=zeros((ny,nx))
# we will add a seed just in case we want to redo this exact phase later...
seed = 20171117
xtopo, ytopo, h, hband, hlow, k, l, P0, Pband, Plow = getTopo2D(
        dx[0], maxx+dx[0],
        dy[0],maxy+dy[0],
        mu=3.5, K0=K0, L0=K0,
       amp=amp, kmax=1./300., kmin=1./6000., seed=seed)
_log.info('shape(hlow): %s', np.shape(hlow))
_log.info('maxx %f dx[0] %f maxx/dx %f nx %d', maxx, dx[0], maxx/dx[0], nx)
_log.info('maxxy %f dy[0] %f maxy/dy %f ny %d', maxy, dy[0], maxy/dy[0], ny)
d = np.real(hlow)
d=d-H

with open(indir+"/topog.bin", "wb") as f:
  d.tofile(f)
f.close()

_log.info(shape(d))

fig, ax = plt.subplots(2,1)
_log.info('%s %s', shape(x),shape(d))
ax[0].plot(x/1.e3,d[0,:].T)
pcm=ax[1].pcolormesh(x/1.e3,y/1.e3,d,rasterized=True)
fig.colorbar(pcm,ax=ax[1])
fig.savefig(outdir+'/figs/topo.png')


##################
# dz:
# dz is from the surface down (right?).  Its saved as positive.
dz = ones((1,nz))*H/nz

with open(indir+"/delZ.bin", "wb") as f:
	dz.tofile(f)
f.close()
z=np.cumsum(dz)

####################
# temperature profile...
#
# temperature goes on the zc grid:
g=9.8
alpha = 2e-4
T0 = 28+cumsum(N0**2/g/alpha*(-dz))

with open(indir+"/TRef.bin", "wb") as f:
	T0.tofile(f)
f.close()
#plot
plt.clf()
plt.plot(T0,z)
plt.savefig(outdir+'/figs/TO.pdf')

########################
# RBCS sponge and forcing
# In data.rbcs, we have set tauRelaxT=17h = 61200 s
# here we wil set the first and last 50 km in *y* to relax at this scale and
# let the rest be free.

iny = np.where((y<50e3) | (y>maxy-50e3))[0]

aa = np.zeros((nz,ny,nx))
for i in iny:
    aa[:,:,i]=1.

with open(indir+"/spongeweight.bin", "wb") as f:
    aa.tofile(f)
f.close()

aa=np.zeros((nz,ny,nx))
aa+=T0[:,newaxis,newaxis]
_log.info(shape(aa))

with open(indir+"/Tforce.bin", "wb") as f:
    aa.tofile(f)
f.close()

###### Manually make the directories
#for aa in range(128):
#    try:
#        mkdir(outdir0+'%04d'%aa)
#    except:
#        pass

_log.info('Writing info to README')
############ Save to README
with open('README','r') as f:
  data=f.read()
with open('README','w') as f:
  import datetime
  import time
  ts = time.time()
  st=datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
  f.write( st+'\n')
  f.write( outdir+'\n')
  f.write(comments+'\n\n')
  f.write(data)

_log.info('All Done!')

_log.info('Archiving to home directory')

try:
    shutil.rmtree('../archive/'+runname)
except:
    pass

shutil.copytree(outdir0+'/input/', '../archive/'+runname+'/input')
shutil.copytree(outdir0+'/python/', '../archive/'+runname+'/python')
shutil.copytree(outdir0+'/code', '../archive/'+runname+'/code')

exit()
