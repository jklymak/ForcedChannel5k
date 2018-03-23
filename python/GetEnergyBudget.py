# -*- coding: utf-8 -*-
# In[1]:

#import matplotlib.pyplot as plt
import numpy as np
import  pickle
#import matplotlib.colors as mcolors
import dask as da
import xarray as xr
import xmitgcm as xm
import sys
from dask.diagnostics import ProgressBar
import logging

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

print(len(sys.argv))
print(sys.argv)
if len(sys.argv)>=3:
    pre = sys.argv[1]
    U0  = float(sys.argv[2])
    N0=1e-3
    f0=1.e-4
if len(sys.argv)>=4:
    N0  = float(sys.argv[3])/10000
if len(sys.argv)>=5:
    f0=float(sys.argv[4])/1000000
else:
    sys.exit('GetEnergyBudget.py prefix U0 N0 f0')

print(U0)

submean = True


v0 = np.array([0.,0.,0.])
u0 = np.array([0.,0.,0.])
dss = xr.open_dataset('../results/%s/input/final.nc' % pre,
        chunks={'record':1, 'i': 30, 'i_g': 30})

for i in range(3):
    g=9.8
    alpha = 2e-4
    nz = dss['Z'].size
    _log.debug('nz %d', nz)
    dz=4000/nz
    T0 = 28-np.cumsum(N0**2/g/alpha*dz*np.ones(nz))
    if i == 0:
        dss['T0']=(('k'),T0)

    z=dss['Z']
    if submean:
        v0[i] = (dss['VVEL'].isel(record=i,
                k=((z < -50.) & (z > -2000.))).mean()).values
        u0[i] = (dss['UVEL'].isel(record=i,
                k=((z < -50.) & (z > -2000.))).mean()).values - U0
    else:
        v0[i]=0.
        u0[i]=0.
    dss['VVEL'].values[i, :, :, :] = dss['VVEL'].isel(record=i) - v0[i]
    dss['UVEL'].values[i, :, :, :] = dss['UVEL'].isel(record=i) - u0[i]


energy=[[],[],[]]
for n in range(3):
    energy[n] = dict()
    energy[n]['Z']=dss['Z'].values
    energy[n]['time']=dss['time'][n].values
    energy[n]['drF']=dss['drF'].values
    energy[n]['area']=dss['rA'].sum().values

ds=dss.isel(record=1)

## get dWdP term at each depth.
# get the pressure on the Zl points:
print('Starting d(WP)/dz')
Pp = (ds['PHIHYD'][1:,:,:].data * ds['hFacC'][1:,:,:].data +
        ds['PHIHYD'][:-1,:,:].data * ds['hFacC'][:-1,:,:].data)
print(Pp)
print(type(Pp))
print(Pp)
print(type(Pp))

with ProgressBar():
    Pp = np.divide(Pp, (ds['hFacC'][1:,:,:].data + ds['hFacC'][:-1,:,:].data))
    WP=(( Pp * ds['WVEL'][1:,:,:] * ds['rA']).sum(dim=('j', 'i'))).values
dWPz = np.diff(WP,axis=0)/ds['drF'][1:-1]

dWPz = np.hstack((0, dWPz, 0))
energy[1]['dWPdz'] = -dWPz

print(energy[1].keys())
print('Starting Body Force')
## Get the body force....
# clean up...
## sizes wrong here....
V = xr.DataArray(0.5*(ds['VVEL'] + ds.VVEL.roll(j_g=1)).data,
        dims=('k','j','i'))

xx=(V * ds['hFacC'] * f0 * U0 * ds['rA']).sum(dim=('j','i'))
with ProgressBar():
    energy[1]['Bf'] = xx.data

print(energy[1]['Bf'])

## KE and PE:
if 1:
    for n in range(3):
        ds = dss.isel(record=n)
        print('Starting KE')

        with ProgressBar():
            xx = (0.5*(ds['UVEL']**2 * ds['hFacW'].data) *
                    ds['rA'].data
                 ).sum(dim=('j','i_g')).compute()
            xx += (0.5*(ds['VVEL']**2 * ds['hFacS'].data) *
                   ds['rA'].data
                  ).sum(dim=('j_g','i')).compute()
        energy[n]['KE']=xx
        print(energy[n].keys())
        with ProgressBar():
            energy[n]['PE'] = (g * (4e-4 * (ds['THETA'] - ds['T0']))**2
                               / 2. / N0**2 *
                               ds['hFacC']*ds['rA']).sum(dim=('i','j')
                              ).data
        print(energy[n].keys())

    print('Done KE and PE')

## Epsilon:
# should I be saving epsilon?
if 0:
    print('Starting epsilon')
    ds = dss.isel(record=1)
    ds['KLepsZ']=xr.DataArray(ds['KLeps'].data,dims=('time','Z','YC','XC'),
                              coords={'Z':ds['Z'].values})

    with ProgressBar():
        energy[1]['eps'] = (ds['KLepsZ']*ds['hFacC']*ds['rA']).sum(dim=('XC','YC')).compute()
    print(energy[1].keys())


if 0:

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

# WRITE as a dumb pickle
name='../reduceddata/EnergyDemean'+pre+'.pickle'
print('Writing '+name)
with open(name,'wb') as f:
    pickle.dump(energy,f)
# make and write a netcdf
dEdt = energy[-1]['KE'] - energy[0]['KE']
dEdt += (energy[-1]['PE'] - energy[0]['PE'])*9.8/4.0
dEdt = dEdt/3600.

resid = -dEdt+energy[1]['Bf']-energy[1]['dWPdz']
ennc = xr.Dataset(data_vars={'Bf': (['Z'], energy[1]['Bf']),
                            'dWPdz': (['Z'], energy[1]['dWPdz']),
                            # 'eps': (['Z'], energy[1]['eps'][0]),
                            'dEdt': (['Z'], dEdt),
                            'resid': (['Z'], resid),
                            'area': (energy[1]['area'])
                            },
                 coords={'Z': (energy[1]['Z'])})
name='../reduceddata/EnergyDemean'+pre+'.nc'
ennc.to_netcdf(name)
# try to copy over to valdez
# bah doesnt work :-(
# subprocess.call(['scp', name, 'valdez:AbHillParam/reduceddata')
