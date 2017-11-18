import numpy as np
import logging

_log = logging.getLogger(__name__)

"""
Make some topography.  I'm doing this in a separate step becuase for runs with
full spectrum versus narrow spectrum, it would be nice if it was the
same narrow spectrum topography (i.e. one was just the filtered version of
the other.

Want this to be order 400x120 km.

So I think we want 128*32 in x = 4096 and 74*16 in y = 1184
"""


def getTopo2D(dx, maxx, dy, maxy,
              mu=3.5, K0=1.8e-4/2./np.pi, L0=1.8e-4/2./np.pi,
              amp=155., kmax=1./300., kmin=1./6000., seed=None):

    k = np.arange(-1./dx/2., 1./dx/2.,1./maxx) # k is cycles per m
    l = np.arange(-1./dy/2., 1./dy/2.,1./maxy) # k is cycles per m
    x = np.arange(0.,maxx,dx)
    y = np.arange(0.,maxy,dy)
    _log.debug('Len(x) %d', len(x))
    _log.debug('Len(k) %d', len(k))
    _log.debug('Len(y) %d', len(y))
    _log.debug('Len(l) %d', len(l))

    dk = 1./maxx
    dl = 1./maxy
    _log.debug((kmin,kmax))
    _log.debug((dk))
    Nx = maxx/dx
    Ny = maxy/dy

    K,L = np.meshgrid(k,l)
    _log.debug((np.shape(K)))
    _log.debug((np.shape(k)))
    twopi=2.*np.pi

    N = 2*len(k)+1
    # 2-D powerspectrum
    P = (mu-2)*(amp**2)*twopi/(K0*twopi)/(L0*twopi) * (1.+ (K/K0)**2+(L/L0)**2)**(-mu/2)
    _log.debug((np.shape(P)))
    _log.debug((np.shape(K)))
    KK = np.sqrt(K**2+L**2)
    P0 = P.copy()

    np.random.seed(seed)
    phase = np.random.rand(np.shape(K)[0], np.shape(K)[1] )*twopi

    for ii in range(3):
        P = P0.copy()
        if ii ==1:
            P[KK<kmin]=1e-10
            P[KK>kmax]=1e-10
        if ii==2:
            P[KK>kmin]=1e-10
        A = np.sqrt(maxx*maxy*P/2.)*np.exp(1j*phase)
        # now build a matrix to ifft:
        # this is why we had to divide by 2 above....
        A = np.fft.fftshift(A,axes=0)
        A = np.fft.fftshift(A,axes=1)
        #A = np.vstack((A,np.conj(A[::-1,:])))
        A = np.vstack((0.*A[[0],:],A))
        A = np.hstack((0.*A[:,[0]],A))
        _log.debug((np.shape(A)))
        if ii==0:
            h = np.fft.ifft2(A)/dx/dy
        if ii==1:
            hband = np.fft.ifft2(A)/dx/dy
            Pband = P.copy()
        if ii==2:
            hlow = np.fft.ifft2(A)/dx/dy
            Plow = P.copy()


    return x,y,h[:-1,:-1],hband[:-1,:-1],hlow[:-1,:-1],k,l,P0,Pband,Plow


if __name__ == "__main__":
    dx0 = 100.
    dy0 = 100.
    maxx = 409600
    maxy = 118400

    xh,yh,h,hband,hlow,k,l,P,Pband,Plow=getTopo2D(dx0,maxx,dy0,maxy,amp=305.,kmin=1./6000.,kmax=1./300.)

    # h0 is the full spectrum
    # h is between kmin and kmax
    # massage some:
    h = np.real(h - np.min(h)).astype(dtype='int16')
    hband = np.real(hband - np.mean(hband)+np.mean(h)).astype(dtype='int16')
    hlow = np.real(hlow - np.mean(hlow)+np.mean(h)).astype(dtype='int16')



    with open("../indata/h.pickle","wb") as f:
        pickle.dump({'h':h,'xh':xh,'yh':yh},f)
    with open("../indata/hlow.pickle","wb") as f:
        pickle.dump({'h':hlow,'xh':xh,'yh':yh},f)
    with open("../indata/hband.pickle","wb") as f:
        pickle.dump({'h':hband,'xh':xh,'yh':yh},f)
