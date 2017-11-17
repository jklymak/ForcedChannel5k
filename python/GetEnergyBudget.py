# -*- coding: utf-8 -*-
# In[1]:

#import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
#import matplotlib.colors as mcolors
import xarray as xr
import xmitgcm as xm
import sys
from dask.diagnostics import ProgressBar

print(len(sys.argv))
if len(sys.argv)==4:
    pre = sys.argv[1]
    U0  = float(sys.argv[2])
    first = int(sys.argv[3])
else:
    sys.exit('GetEnergyBudget.py prefix U0 last')

print pre

N0=1e-3
g=9.8
alpha = 2e-4
dz=10.
T0 = 28-np.cumsum(N0**2/g/alpha*dz*np.ones(400))

dss=[0,0,0]
iters = np.array([-1440,-720,0])+first
print(iters)
for i in range(3):
    n=i
    dss[i] = xr.open_dataset('../results/'+pre+'/_Model/input/ds%010d.nc'%iters[i],
                          chunks={'Z':20,'Zl':20})
    dss[i]['T0']=(('Z'),T0)
    indy = np.where(np.abs(dss[n]['dyG'][:,0]-100.)<0.001)[0]
    indx = np.where(np.abs(dss[n]['dxG'][0,:]-100.)<0.001)[0]
    #    indx = np.where(np.abs(dss[n]['X']-np.mean(dss[n]['X']))<34e3)[0]
    dss[n]= dss[n].isel(YG=indy,YC=np.hstack((indy[0]-1,indy)),
                        XG=indx,XC=np.hstack((indx[0]-1,indx)))
    print(dss[n]['time'])
# this grid has one extra YC and XC bracketing the XG and YG variables...
# We will operate on the inner ring of XC,YC.

# get kinetic energy at center of grid cells
#
# integration is at ind = 1 to ind = -1.  This allows us to get data from the U,V cells at those 
# points.
#
# What we really should have done was actualy subset P one larger than U (the opposite of the
# usual).  

energy=[[],[],[]]
for n in range(3):
    energy[n] = dict()
    energy[n]['Z']=dss[n]['Z'].values
    energy[n]['time']=dss[n]['time'].values
    energy[n]['drF']=dss[n]['drF'].values

ia = slice(1,-1) # should be inner data (excluding edges)
a  = slice(None,-1) # should be first to second last
b  = slice(1,None) # should be second to last
ds=dss[1]

## get dWdP term at each depth. 
# get the pressure on the Zl points:
print('Starting d(WP)/dz')
PL = xr.DataArray(0.5*(ds['PH'][:,1:,:,:].data+ds['PH'][:,:-1,:,:].data),dims=('time','Zl','YC','XC'),
                        coords={'Zl':ds['Zl'][1:].values,'YC':ds['YC'].values,'XC':ds['XC'].values })

with ProgressBar():
    WP=((PL*ds['W']*ds['rA']).sum(dim=('YC','XC'))).values
dWPz = np.hstack((np.zeros((1,1)),np.diff(WP,axis=1)/ds['drF'].values[np.newaxis,1:-1],np.zeros((1,1))))
energy[1]['dWPdz'] = -dWPz

print(energy[1].keys())


print('Starting Body Force')
## Get the body force.... 
f0=1e-4
# clean up...
## sizes wrong here....
print(ds['V'])
ds['VC'] = xr.DataArray(0.5*(ds['V'].isel(YG=a).data+
                             ds['V'].isel(YG=b).data),
                        dims=('time','Z','YC','XC'),
                        coords={'YC':ds['YC'][1:-1]})
print(ds['VC'])
xx=((ds['VC'].isel(YC=ia,XC=ia))*ds['hFacC'].isel(YC=ia,XC=ia)
    *f0*U0*ds['rA'].isel(XC=ia,YC=ia)).sum(dim=('XC','YC'))
with ProgressBar():
    energy[1]['Bf'] = xx.data.compute()
    


## KE and PE:
if 1:
    for n in range(3):
        print('Starting KE')
        
        with ProgressBar():
            xx = (0.5*0.5*(dss[n]['U'].isel(XG=a,YC=ia)**2 * dss[n]['hFacW'].isel(XG=a,YC=ia).data) *
                  dss[n]['rA'].isel(XC=ia,YC=ia).data).sum(dim=('YC','XG')).compute()
            xx += (0.5*0.5*(dss[n]['U'].isel(XG=b,YC=ia)**2 * dss[n]['hFacW'].isel(XG=b,YC=ia).data) *
                   dss[n]['rA'].isel(XC=ia,YC=ia).data).sum(dim=('YC','XG')).compute()
            xx += (0.5*0.5*(dss[n]['V'].isel(YG=a,XC=ia)**2 * dss[n]['hFacS'].isel(YG=a,XC=ia).data) *
                   dss[n]['rA'].isel(XC=ia,YC=ia).data).sum(dim=('YG','XC')).compute()
            xx += (0.5*0.5*(dss[n]['V'].isel(YG=b,XC=ia)**2 * dss[n]['hFacS'].isel(YG=b,XC=ia).data) *
                   dss[n]['rA'].isel(XC=ia,YC=ia).data).sum(dim=('YG','XC')).compute()
        energy[n]['KE']=xx
        print(energy[n].keys())
        with ProgressBar():
            energy[n]['PE'] = (g*(4e-4*(dss[n]['T'].isel(XC=ia,YC=ia)-dss[n]['T0']))**2/2./N0**2*dss[n]['hFacC'].isel(XC=ia,YC=ia)*dss[n]['rA'].isel(XC=ia,YC=ia)).sum(dim=('XC','YC')).data.compute()
        print(energy[n].keys())

    print('Done KE and PE')

## Epsilon:
print('Starting epsilon')
ds['KLepsZ']=xr.DataArray(ds['KLeps'].data,dims=('time','Z','YC','XC'),
                          coords={'Z':ds['Z'].values})

with ProgressBar():
    energy[1]['eps'] = (ds['KLepsZ']*ds['hFacC']*ds['rA']).isel(XC=ia,YC=ia).sum(dim=('XC','YC')).compute()
print(energy[1].keys())



    
print('Starting linear internal wave fluxes')

# upW: This is on the LHS.  Need PH on the G face...
hf = ds['hFacC'].isel(XC=0)+ds['hFacC'].isel(XC=1)
p = (ds['PH'].isel(XC=0,YC=ia)*ds['hFacC'].isel(XC=0,YC=ia)+ds['PH'].isel(XC=1,YC=ia)*ds['hFacC'].isel(XC=1,YC=ia))/hf
up = p*ds['U'].isel(XG=0,YC=ia)
with ProgressBar():
    xx=(up*up['dyG'].isel(YC=ia)).sum(dim='YC').data.compute()
    energy[1]['upW'] = xx


# upE
hf = ds['hFacC'].isel(XC=-2)+ds['hFacC'].isel(XC=-1)
p = (ds['PH'].isel(XC=-2,YC=ia)*ds['hFacC'].isel(XC=-2,YC=ia)+ds['PH'].isel(XC=-1,YC=ia)*ds['hFacC'].isel(XC=-1,YC=ia))/hf
up = p*ds['U'].isel(XG=-1,YC=ia)
with ProgressBar():
    energy[1]['upE'] = (up*up['dyG'].isel(YC=ia)).sum(dim='YC').data.compute()


# vpS
hf = ds['hFacC'].isel(YC=(0,1)).sum(dim='YC')
p = (ds['PH'].isel(YC=(0,1),XC=ia)*ds['hFacC'].isel(YC=(0,1),XC=ia)).sum(dim='YC')/hf
up = p*ds['V'].isel(YG=0,XC=ia)
with ProgressBar():
    energy[1]['vpS'] = ((up*up['dxG'].isel(XC=ia)).sum(dim='XC')).data.compute()

# vpN
hf = ds['hFacC'].isel(YC=(-2,-1)).sum(dim='YC')
p = (ds['PH'].isel(YC=(-2,-1),XC=ia)*ds['hFacC'].isel(YC=(-2,-1),XC=ia)).sum(dim='YC')/hf
up = p*ds['V'].isel(YG=-1,XC=ia)
with ProgressBar():
    energy[1]['vpN'] = ((up*up['dxG'].isel(XC=ia)).sum(dim='XC')).data.compute()

print('Done linear energy fluxes')
print(energy[1])

print('start non-linear energy fluxes')


print('veS')
# veS: South side non-linear energy flux...
xx = 0 # for the XG variable, this is what we want (U)
xs = (0,1) # for the XC variables, we need to average these two (V)
E = 0.5*ds['V'].isel(YG=xx,XC=ia)**2
# need to average 4 V points around each U:
Ev = 0.5*(ds['U'].isel(YC=xs,XG=a)**2*ds['hFacW'].isel(YC=xs,XG=a)).sum(dim='YC').data
Ev += 0.5*(ds['U'].isel(YC=xs,XG=b)**2*ds['hFacW'].isel(YC=xs,XG=b)).sum(dim='YC').data
dz =  (ds['hFacW'].isel(YC=xs,XG=a)).sum(dim='YC').data
dz += (ds['hFacW'].isel(YC=xs,XG=b)).sum(dim='YC').data
E += Ev/dz
# APE:
PE = (0.5*g*(4.e-4*(ds['T'].isel(YC=xs,XC=ia)-ds['T0']))**2/N0**2*ds['hFacC'].isel(YC=xs,XC=ia)).sum(dim='YC')
E += PE/(ds['hFacC'].isel(YC=xs,XC=ia)).sum(dim='YC')
with ProgressBar():
    energy[1]['veS'] = (E*ds['V'].isel(YG=xx,XC=ia)*ds['dxG'].isel(YG=xx,XC=ia)).sum(dim='XC').data.compute()

print(energy[1])

print('veN')
# veN: North side non-linear energy flux...
xx = -1 # for the XG variable, this is what we want (U)
xs = (-2,-1) # for the XC variables, we need to average these two (V)
E = 0.5*ds['V'].isel(YG=xx,XC=ia)**2
# need to average 4 V points around each U:
Ev = 0.5*(ds['U'].isel(YC=xs,XG=a)**2*ds['hFacW'].isel(YC=xs,XG=a)).sum(dim='YC').data
Ev += 0.5*(ds['U'].isel(YC=xs,XG=b)**2*ds['hFacW'].isel(YC=xs,XG=b)).sum(dim='YC').data
dz =  (ds['hFacW'].isel(YC=xs,XG=a)).sum(dim='YC').data
dz += (ds['hFacW'].isel(YC=xs,XG=b)).sum(dim='YC').data
E += Ev/dz
# APE:
PE = (0.5*g*(4.e-4*(ds['T'].isel(YC=xs,XC=ia)-ds['T0']))**2/N0**2*ds['hFacC'].isel(YC=xs,XC=ia)).sum(dim='YC')
E += PE/(ds['hFacC'].isel(YC=xs,XC=ia)).sum(dim='YC')
with ProgressBar():
    energy[1]['veN'] = (E*ds['V'].isel(YG=xx,XC=ia)*ds['dxG'].isel(YG=xx,XC=ia)).sum(dim='XC').data.compute()

print('ueE')
# uEE: West side non-linear energy flux...
xx = -1 # for the XG variable, this is what we want (U)
xs = (-2,-1) # for the XC variables, we need to average these two (V)
E = 0.5*ds['U'].isel(XG=xx,YC=ia)**2
# need to average 4 V points around each U:
Ev = 0.5*(ds['V'].isel(XC=xs,YG=a)**2*ds['hFacS'].isel(XC=xs,YG=a)).sum(dim='XC').data
Ev += 0.5*(ds['V'].isel(XC=xs,YG=b)**2*ds['hFacS'].isel(XC=xs,YG=b)).sum(dim='XC').data
dz =  (ds['hFacS'].isel(XC=xs,YG=a)).sum(dim='XC').data
dz += (ds['hFacS'].isel(XC=xs,YG=b)).sum(dim='XC').data
E += Ev/dz
# APE:
PE = (0.5*g*(4.e-4*(ds['T'].isel(XC=xs,YC=ia)-ds['T0']))**2/N0**2*ds['hFacC'].isel(XC=xs,YC=ia)).sum(dim='XC')
E += PE/(ds['hFacC'].isel(XC=xs,YC=ia)).sum(dim='XC')
with ProgressBar():
    energy[1]['ueE'] = (E*ds['U'].isel(XG=xx,YC=ia)*ds['dyG'].isel(XG=xx,YC=ia)).sum(dim='YC').data.compute()

# uEW: West side non-linear energy flux...
E = 0.5*ds['U'].isel(XG=0,YC=ia)**2
# need to average 4 V points around each U:
Ev = 0.5*(ds['V'].isel(XC=(0,1),YG=a)**2*ds['hFacS'].isel(XC=(0,1),YG=a)).sum(dim='XC').data
Ev += 0.5*(ds['V'].isel(XC=(0,1),YG=b)**2*ds['hFacS'].isel(XC=(0,1),YG=b)).sum(dim='XC').data
dz =  (ds['hFacS'].isel(XC=(0,1),YG=a)).sum(dim='XC').data
dz += (ds['hFacS'].isel(XC=(0,1),YG=b)).sum(dim='XC').data
E += Ev/dz
# APE:
PE = (0.5*g*(4.e-4*(ds['T'].isel(XC=(0,1),YC=ia)-ds['T0']))**2/N0**2*ds['hFacC'].isel(XC=(0,1),YC=ia)).sum(dim='XC')
E += PE/(ds['hFacC'].isel(XC=(0,1),YC=ia)).sum(dim='XC')
with ProgressBar():
    energy[1]['ueW'] = (E*ds['U'].isel(XG=0,YC=ia)*ds['dyG'].isel(XG=0,YC=ia)).sum(dim='YC').data.compute()

print(energy[1])

# WRITE!
name='Energy'+pre+'%010d.pickle'%iters[1]
print('Writing '+name)
with open(name,'w') as f:
    pickle.dump(energy,f)



