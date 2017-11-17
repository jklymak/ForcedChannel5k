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

runname='lee3dfull01'
outdir='/scr/jklymak/scratch/'+runname+'/'
basep='../results/'+runname+'/'
dn = 720
timestd = dn*np.array([42000/dn])
timestd = [71280-720]
#timestd = [0]

try:
    mkdir(outdir)
except:
    pass

oldfiles=[]

dz = rdmds(basep + '_Model/input/'+'DRF',machineformat='l');
z = squeeze(rdmds(basep + '_Model/input/'+'RC',machineformat='l')[:,0,0]);
zf = rdmds(basep + '_Model/input/'+'RF',machineformat='l');
#T = rdmds(basep + '0*/T',num,machineformat='l',fill_value=numpy.NaN)
#x = rdmds(basep + '0*/XC',machineformat='l',fill_value=numpy.NaN)
xc = squeeze(rdmds(basep + '0*/XC',machineformat='l',fill_value=numpy.NaN)[0,:])
x0 = numpy.mean(xc)
xc=xc-x0
inx = np.where(np.abs(xc)<50.e3)[0]
print inx
print('inx %d %d'%(inx[0],inx[-1]))
x=xc
yc = squeeze(rdmds(basep + '0*/YC',machineformat='l',fill_value=numpy.NaN)[:,0])
y=yc
ny = np.shape(y)[0]
print('Ny=%d'% ny)

nx0=shape(x)[0]
xg = squeeze(rdmds(basep + '0*/XG',machineformat='l',fill_value=numpy.NaN)[0,:])
xg=xg-x0
dxc = squeeze(rdmds(basep + '0*/DXC',machineformat='l',fill_value=numpy.NaN)[0,:])
#dxg = squeeze(rdmds(basep + '0*/DXG',machineformat='l',fill_value=numpy.NaN)[0,:])

N=shape(x)[0]
nx = shape(x)[0]
print nx

x = xc/1000.
y=yc/1000.


#hFacC=rdmds(basep + '0*/hFacC',machineformat='l',fill_value=numpy.NaN)
Dep = rdmds(basep+'0*/Depth',machineformat='l')
#for num in range(72000,86800,400):

ind = 5

reg=(inx[0],inx[-1],0,ny)
print reg


for timeind in timestd:
#for timeind in [780]:

    num=timeind
    tds = ['T','U','V','W','Eta','PH','KLeps-T']
    for td in tds:
	D=dict()
        D[td] = squeeze(rdmds(basep + '0*/'+td,num,region=reg,machineformat='l',fill_value=numpy.NaN))
	D['Depth']=Dep[0]
	D['x']=x
	D['y']=y
	D['z']=z
	D['dzc']=dz
	with open(outdir+'Snap%s%05d.pickle'%(td,timeind),'wb') as f:
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
            T = rdmds(basep + '0*/T',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            PH = rdmds(basep + '0*/PH',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            PNH = rdmds(basep + '0*/PNH',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            T=T[:,0,:]
            U = rdmds(basep + '0*/U',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            U=U[:,0,:]
            V = rdmds(basep + '0*/V',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            V=V[:,0,:]
            W = rdmds(basep + '0*/W',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            W=W[:,0,:]
            Eta = rdmds(basep + '0*/Eta',num,machineformat='l',region=reg,fill_value=numpy.NaN)
            savez(outdir+'slice%010d' % num,T=T,U=U,V=V,W=W,Eta=Eta,z=z,x=x,Dep=Dep,hFacC=hFacC,xc=xc,PH=PH,PNH=PNH,dxc=dxc,dz=dz)
        if 1:
            for diag in diags:
                dat=rdmds(basep + '0*/'+diag,num,machineformat='l',fill_value=numpy.NaN)[0]
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

