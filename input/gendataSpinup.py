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
from replace_data import replace_data
import xarray as xr


def bindata(binx,biny,x,y,X):
    """
    bindata(binx,biny,x,y,X):
    returns data Xnew binned into binx and biny bins
    meanX,varX,nX=bindata(binx,biny,x,y,X)
    """

    import numpy as np

    M = size(biny,0)-1
    N = size(binx,0)-1

    good = ~np.isnan(x+y+X)
    x=x[good]
    y=y[good]
    X=X[good]

    meanX = np.zeros((M,N))
    varX = np.zeros((M,N))
    nX = np.zeros((M,N))

    #print shape(binx)
    #print shape(arange(0,N,1.0))

    indx = floor(np.interp(x,binx,arange(0,N+1,1.0))).astype(np.int16)
    indx[indx<0]=0
    indx[indx>N-1]=N-1

    indy = floor(np.interp(y,biny,arange(0,M+1,1.0))).astype(np.int16)
    #print shape(X)
    indy[indy<0]=0
    indy[indy>M-1]=M-1
    for ind in range(len(X)):
        meanX[indy[ind],indx[ind]]+=X[ind]
        nX[indy[ind],indx[ind]]+=1
    nX[nX==0.0]=NaN;
    meanX = meanX/nX;

    for ind in range(len(x)):
        varX[indy[ind],indx[ind]]+=(meanX[indy[ind],indx[ind]]-X[ind])**2
    varX=varX/nX
    # mask
    meanX=np.ma.masked_where(isnan(nX),meanX)
    nX=np.ma.masked_where(isnan(nX),nX)
    varX=np.ma.masked_where(isnan(nX),varX)


    return meanX,varX,nX


logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger(__name__)

runtype = 'lindrag'  # 'full','filt','low'
runname='Channel5k1000_%s_01' % runtype
comments="""5k -re-spinup w smooth topo but a linear drag to emulate based on my
            parameters"""

# to change U we need to edit external_forcing recompile

outdir0='../results/'+runname+'/'

indir =outdir0+'/indata/'

## Params for below as per Nikurashin and Ferrari 2010b
H = 3440.
H0 = 3000

# model size

nx = 40*8
ny = 26*8
nz = 84
dx0 = 1600e3/nx
dy0 = 1040e3/ny

_log.info('nx %d ny %d', nx, ny)


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

try:
    shutil.copy('../build/mitgcmuv', outdir+'/../build/mitgcmuv')
    #shutil.copy('../build/mitgcmuvU%02d'%u0, outdir+'/../build/mitgcmuv%02d'%u0)
    shutil.copy('../build/Makefile', outdir+'/../build/Makefile')
    shutil.copy('dataSpunupLinDrag', outdir + '/dataSpunupLinDrag')
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
except:
    _log.warning('Not copying files for some reason')

_log.info("Done copying files")

####### Make the grids #########

# Make grids:

##### Dx ######

dx = zeros(nx)+dx0
print(len(dx))

# dx = zeros(nx)+100.
x=np.cumsum(dx)
x=x-x[0]
maxx=np.max(x)
_log.info('XCoffset=%1.4f'%x[0])

##### Dy ######

dy = zeros(ny)+dy0

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
h0 = 1000
sig = 75e3

d = d + h0 * np.exp(-((x-x.mean())/sig)**2)[np.newaxis, :]
d = -H0 + d
# now do the edges:
dedge = np.zeros((ny, nx)) - H0
ind = np.where(y<50e3)[0]
dedge[ind,:] = -((y[ind])*(H0)/50e3)[:, np.newaxis]
dedge[-ind,:] = -(((y[-1]-y[-ind]))*(H0)/50e3)[:, np.newaxis]

d[dedge>d] = dedge[dedge>d]
d[0, :] = 0
d0 = d

hlow = np.zeros((ny, nx))
with xr.open_dataset('../indata/topo1kSpread.nc') as topods:
    # the data set is 1-km, and we want on this grid so bin...
    X, Y = np.meshgrid(topods.x.data, topods.y.data)
    xx = np.hstack((x, x[-1]+dx[-1]))
    yy = np.hstack((y, y[-1]+dy[-1]))
    hlow, n, v = bindata(xx, yy, X.flatten(), Y.flatten(), topods.hlow.data.flatten())

hlow = hlow.data

fig, ax = plt.subplots()
ax.pcolormesh(x, y, hlow, rasterized=True)
fig.savefig(outdir + '/figs/topolow.png')



#
# put in an envelope:

sig = 300e3
xenvelope = np.zeros(nx) + 0.07 + 0.93* np.exp(-((x-x.mean())/sig)**2)
xenvelope[np.abs(x-x.mean())<200e3] = 1.
xenvelope = xenvelope * 0

# hband = np.real(hband - np.mean(hband)+np.mean(h))
hlow = np.real(hlow - np.mean(hlow))
hlow = hlow * (xenvelope)[np.newaxis, :]

d = hlow + d

d[0, :] = 0
d[d>0] = 0
d[d<-H] = -H

_log.info(shape(d))

fig, ax = plt.subplots(2,1)
_log.info('%s %s %s %s %s', nx, ny, shape(x),shape(y),shape(d))
mid = int(ny/2)
ax[0].plot(x/1.e3,d[mid,:].T)
ax[0].plot(x/1.e3,d0[mid,:].T)
ax[0].plot(x/1.e3,hlow[mid,:].T)
ax[0].plot(x/1.e3,xenvelope*1000-H)

pcm=ax[1].pcolormesh(x/1.e3,y/1.e3,d,rasterized=True)
fig.colorbar(pcm,ax=ax[1])
fig.savefig(outdir+'/figs/topo.png')

with open(indir+"/topog.bin", "wb") as f:
  d.tofile(f)
f.close()



##################
# dz:
# dz is from the surface down (right?).  Its saved as positive.
dz = ones((1,nz))*H/nz

with open(indir+"/delZ.bin", "wb") as f:
	dz.tofile(f)
f.close()
z=np.cumsum(dz)

#######################
# surface temperature relaxation
aa = np.zeros((ny, nx))
aa = aa + np.linspace(4, 12, ny)[:, np.newaxis]
with open(indir+"/thetaClimFile.bin", "wb") as f:
	aa.tofile(f)

fig, ax = plt.subplots()
ax.plot(aa[:,0], y / 1e3)
ax.set_title('Surface temperature relaxation')
ax.set_xlabel('T [degC]')
ax.set_ylabel('y [km]')
fig.savefig(outdir + '/figs/Tsurf.png')

#######################
# surface zonalWindFile
aa = np.zeros((ny, nx))
tau0 = 0.2 # N/m^2
tauoffset = 0.0
windwidth = 1040e3
tau = tau0 * np.cos((y-y.mean())/ windwidth * np.pi )**2 + tauoffset
aa = aa + tau[:, np.newaxis]
with open(indir+"/zonalWindFile.bin", "wb") as f:
	aa.tofile(f)

fig, ax = plt.subplots()
ax.plot(aa[:,0], y / 1e3)
ax.set_title('Surface temperature relaxation')
ax.set_xlabel(r'$\tau [N\.m^{-2}]$')
ax.set_ylabel('y [km]')
fig.savefig(outdir + '/figs/windSurf.png')

fname = 'Channel5k_5y_Spinup.nc'
fname2d = 'Channel5k_5y_Spinup2d.nc'
_log.info('Reading initial conditions from from {} and {}', fname, fname2d)


####################
# temperature profile...
# surface temperature is going to be from 4 to 12 degrees. Lets make the
# reference temperature 5 degrees.
_log.info('Doing surface height interpolation')
# get data

with xr.open_dataset(fname2d) as ds:
    _log.info('Time', ds.time)
    ny0 = ds.sizes['j']
    nx0 = ds.sizes['i']
    _log.info('nx0, ny0', nx0, ny0)
    # interpolate first onto new x...
    tmp = np.zeros((ny0, nx))
    print(np.shape(ds.ETAN.data))
    for j in range(ny0):
        good = np.isfinite(ds.ETAN.data[j, :])
        xx = ds.XC.data[0, good]
        tmp[j, :] = np.interp(x, xx, ds.ETAN.data[j, good] )

    aa = np.zeros((ny,nx))
    # now interpolate in y....
    for i in range(nx):
        good = np.isfinite(tmp[:, i])
        aa[:, i] = np.interp(y, ds.YC.data[good, 0], tmp[good, i])
    with open(indir+"/Etainit.bin", "wb") as f:
        aa.tofile(f)

    fig, ax = plt.subplots(2, 1)
    ax[0].pcolormesh(ds.XC, ds.YC, ds.ETAN, rasterized=True)
    ax[1].pcolormesh(x, y, aa, rasterized=True)
    fig.savefig(outdir+'/figs/Eta0.png')

# do the velocities....
# these are written row-major so I think we can do this by level...
with xr.open_dataset(fname) as dss:
    ny0 = ds.sizes['j']
    nx0 = ds.sizes['i']
    nz0 = ds.sizes['k']
    # get T0
    T0 = dss['THETA'].isel(j=slice(5,-5)).mean(dim=('i', 'j'))
    print(T0)
    dsnew = xr.Dataset( {'UVEL': (['z','y','x'], np.zeros((nz, ny0, nx0))),
                        'VVEL': (['z','y','x'], np.zeros((nz, ny0, nx0))),
                        'THETA': (['z','y','x'], np.zeros((nz, ny0, nx0)))},
                        coords={'z':-z, 'y':dss.YC.data[:,0], 'x':dss.XC.data[0, :]})

    for todo in ['UVEL', 'VVEL', 'THETA']:
        aa = np.zeros((nz, ny0, nx0))
        for j in range(ny0):
            print(j)
            for i in range(nx0):
                good = np.where((dss['THETA'].data[:, j, i]>0))[0]
                if len(good) > 0:
                    #print(dss[todo].data[good, j, i])
                    a = np.interp(z-z[0]/2,
                            -dss['Z'].data[good],
                            dss[todo].data[good, j, i])
                    dsnew[todo][:, j, i] = a

                    if j==20 and i==20:
                        print(dss['Z'].data[good])
                        print(-z)
                        print(a)
                        print(dsnew[todo][:, j, i])
                elif todo == 'THETA':
                    good = np.where(T0>0)[0]
                    dsnew[todo][:, j, i] = np.interp(z-z[0]/2,
                            -dss['Z'].data[good],
                            T0[good])
                if todo == 'THETA':
                    assert np.all(dsnew[todo][:, j, i] > 0)
    print(dsnew['UVEL'][:, 20, 20])
    dsnew.to_netcdf('Zinterp.nc', 'w')

with xr.open_dataset('Zinterp.nc') as dss:
    ny0 = dss.sizes['y']
    nx0 = dss.sizes['x']
    nz0 = dss.sizes['z']
    for k in range(nz0):
        ds = dss.isel(z=k)
        if k==0:
            mode='wb'
        else:
            mode='ab'

        for todo, outname in zip(['UVEL', 'VVEL', 'THETA'],
                ['Uinit.bin', 'Vinit.bin', 'Tinit.bin']):
            tmp = np.zeros((ny0, nx))
            for j in range(ny0):
                good = np.isfinite(ds[todo].data[j, :])
                xx = ds.x.data[good]
                tmp[j, :] = np.interp(x, xx, ds[todo].data[j, good] )

            aa = np.zeros((ny,nx))
            # now interpolate in y....
            for i in range(nx):
                good = np.isfinite(tmp[:, i])
                aa[:, i] = np.interp(y, ds.y.data[good], tmp[good, i])

            with open(indir+outname, mode) as f:
                aa.tofile(f)

########################
# RBCS sponge and forcing
# In data.rbcs, we have set tauRelaxT=17h = 61200 s
# here we wil set the first and last 50 km in *y* to relax at this scale and
# let the rest be free.
if 0:

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
