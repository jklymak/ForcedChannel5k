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

runname=str(sys.argv[1])

outdir='/scr/jklymak/scratch/'+runname+'/'
basep='../results/'+runname+'/_Model/input/'
dn = 720
timestd = dn*np.array([42000/dn])
timestd = [70560,71280]
#timestd = [0]

try:
    mkdir(outdir)
except:
    pass

oldfiles=[]

dz = rdmds(basep +'DRF',machineformat='l');
z = squeeze(rdmds(basep +'RC',machineformat='l')[:,0,0]);
zf = rdmds(basep +'RF',machineformat='l');
#T = rdmds(basep + '0*/T',num,machineformat='l',fill_value=numpy.NaN)
#x = rdmds(basep + 'XC',machineformat='l',fill_value=numpy.NaN)
xc = squeeze(rdmds(basep + 'XC',machineformat='l',fill_value=numpy.NaN)[0,:])
x0 = numpy.mean(xc)
xc=xc-x0

x=xc
yc = squeeze(rdmds(basep + 'YC',machineformat='l',fill_value=numpy.NaN)[:,0])
y=yc

nx0=shape(x)[0]
xg = squeeze(rdmds(basep + 'XG',machineformat='l',fill_value=numpy.NaN)[0,:])
xg=xg-x0
dxc = squeeze(rdmds(basep + 'DXC',machineformat='l',fill_value=numpy.NaN)[0,:])
#dxg = squeeze(rdmds(basep + 'DXG',machineformat='l',fill_value=numpy.NaN)[0,:])

N=shape(x)[0]
nx = shape(x)[0]
print nx

x = xc/1000.
y=yc/1000.


#hFacC=rdmds(basep + 'hFacC',machineformat='l',fill_value=numpy.NaN)
Dep = rdmds(basep+'Depth',machineformat='l')
#for num in range(72000,86800,400):

ind = 5

reg=(0,nx,127,128)
print reg


for timeind in timestd:
#for timeind in [780]:

    D=dict()
    num=timeind
    D['T'] = squeeze(rdmds(basep + 'T',num,region=reg,machineformat='l',fill_value=numpy.NaN))
    D['U'] = squeeze(rdmds(basep + 'U',num,region=reg,machineformat='l',fill_value=numpy.NaN))
    D['V'] = squeeze(rdmds(basep + 'V',num,region=reg,machineformat='l',fill_value=numpy.NaN))
    D['W'] = squeeze(rdmds(basep + 'W',num,region=reg,machineformat='l',fill_value=numpy.NaN))
    D['Eta'] = squeeze(rdmds(basep + 'Eta',num,region=reg,machineformat='l',fill_value=numpy.NaN))
    D['EtaAll'] = squeeze(rdmds(basep + 'Eta',num,machineformat='l',fill_value=numpy.NaN))
    try:
        D['Ph'] = squeeze(rdmds(basep + 'PH',num,region=reg,machineformat='l',fill_value=numpy.NaN))
        D['Eps']=squeeze(rdmds(basep + 'KLeps-T',num,region=reg,machineformat='l',fill_value=numpy.NaN))
        D['Ebc']=rdmds(basep + 'Ebc',num,machineformat='l',fill_value=numpy.NaN)
        D['Ebc0']=rdmds(basep + 'Ebc',num-dn,machineformat='l',fill_value=numpy.NaN)
        D['Conv']=rdmds(basep + 'Conv',num,machineformat='l',fill_value=numpy.NaN)
        D['Ebt']=rdmds(basep + 'Ebt',num,machineformat='l',fill_value=numpy.NaN)
        D['Ebt0']=rdmds(basep + 'Ebt',num-dn,machineformat='l',fill_value=numpy.NaN)
        D['uPbc']=rdmds(basep + 'uPbc',num,machineformat='l',fill_value=numpy.NaN)
        D['uEbc']=rdmds(basep + 'uEbc',num,machineformat='l',fill_value=numpy.NaN)
        #D['vPbc']=rdmds(basep + 'vPbc',num,machineformat='l',fill_value=numpy.NaN)[0]
        #D['vEbc']=rdmds(basep + 'vEbc',num,machineformat='l',fill_value=numpy.NaN)[0]
        D['uPbt']=rdmds(basep + 'uPbt',num,machineformat='l',fill_value=numpy.NaN)
        #D['vPbt']=rdmds(basep + 'vPbt',num,machineformat='l',fill_value=numpy.NaN)[0]
    except:
        pass
    D['Depth']=Dep[0]
    D['x']=x
    D['y']=y
    D['z']=z
    D['dzc']=dz
    #D['hFacC']=hFacC

    with open(outdir+'Diags%05d.pickle'%timeind,'wb') as f:
        pickle.dump(D,f)

todir = 'valdez.seos.uvic.ca:leewaves15/'
syscom= 'rsync -av '+outdir+' '+todir+runname
print syscom
ret=system(syscom)
print ret


exit()



ind=squeeze(where(abs(x)<=15))
print ind[0]
reg=(ind[0],ind[-1],0,1)
x=x[ind]



diags=['Ebt','uPbt','uEbt','Ebc','uPbc','uEbc','Conv']
nn=0
D=dict()
for diag in diags:
    D[diag] = zeros((370,nx0))

for num in arange(1200,96000+10,1200):
#for num in [72000]:
    try:
        print num
        if 1:
            T = rdmds(basep + 'T',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            PH = rdmds(basep + 'PH',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            PNH = rdmds(basep + 'PNH',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            T=T[:,0,:]
            U = rdmds(basep + 'U',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            U=U[:,0,:]
            V = rdmds(basep + 'V',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            V=V[:,0,:]
            W = rdmds(basep + 'W',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            W=W[:,0,:]
            Eta = rdmds(basep + 'Eta',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            savez(outdir+'slice%010d' % num,T=T,U=U,V=V,W=W,Eta=Eta,z=z,x=x,Dep=Dep,hFacC=hFacC,xc=xc,PH=PH,PNH=PNH,dxc=dxc,dz=dz)
        if 1:
            for diag in diags:
                dat=rdmds(basep + ''+diag,num,machineformat='l',fill_value=numpy.NaN)[0]
                D[diag][nn,:]=dat
        nn=nn+1
    except Exception, err:
        sys.stderr.write('ERROR: %s\n' % str(err))
# trim
for diag in diags:
    D[diag]=D[diag][0:nn,:]
# save as a pickle
D['dxc']=dxc
D['xc']=xc
D['xg']=xg

f=open(outdir+'Diags.pickle','wb')
pickle.dump(D,f)
f.close()

