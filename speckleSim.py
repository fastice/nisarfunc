#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:16:03 2020

@author: ian
"""
import numpy as np
from threading import Thread
from queue import Queue
import multiprocessing
import sys
import copy
import time
# import psutil
# import os
from scipy import optimize


def whiteNoisePatch(nr, nc, sigma=1.0):
    '''
    Generate a white noise complex samples patch of size nrxnc
    Parameters
    ----------
    nr : int
        number of rows.
    nc : int
        number of columns.
    sigma : float, optional
        Standard deviation. The default is 1.0.

    Returns
    -------
    np array.

    '''
    re = np.random.normal(0, sigma, size=(nr, nc)).astype(np.float32)
    im = np.random.normal(0, sigma, size=(nr, nc)).astype(np.float32)
    noise = np.array(re + 1j * im)
    return noise


def limitBandwidth(patch, bandwidth):
    '''
    Limit bandwidth of a patch to fraction < 1 specified by bandwidth

    Parameters
    ----------
    patch : 2D complex data
        White noise patch.
    bandwidth : float
        Fraction of bandwidth to limit to.
    Returns
    -------
    None.
    '''
    myFilt = np.zeros(patch.shape)
    hwr, hwc = int(patch.shape[0]/2), int(patch.shape[1]/2)
    bwr, bwc = int(hwr * bandwidth), int(hwc * bandwidth)
    myFilt[hwr - bwr:hwr + bwr - 1, hwc - bwc:hwc + bwc - 1] = 1
    fftForward = np.fft.fftshift(np.fft.fft2(patch)) * myFilt
    # inverse transform
    patch[:] = np.fft.ifft2(np.fft.ifftshift(fftForward))


def correlatedShiftedPatches(nr, nc, dr, dc, rho, overSample=7, sigma=1.0,
                             bandwidth=1.):
    '''
    Parameters
    ----------
    nr : int
        number of rows.
    nc : int
        number of columns.
    overSample : int, optional
        Oversample factor. The default is 7.
    sigma : float, optional
        Standard deviation. The default is 1.0..
    Returns
    -------
    patch1 : 2D complex data
        Reference patch.
    patch2 : 2D complex data
        Shifted and decorrelated patch.
    '''
    # compute scale factor for rho
    gamma = rho**2 / (2. * rho**2 - 1)
    alpha = gamma - np.sign(gamma) * np.sqrt(gamma**2 - gamma)
    # first patch
    patch1 = whiteNoisePatch(nr, nc, sigma=sigma)
    # second patch
    decorrNoise = whiteNoisePatch(nr, nc, sigma=sigma)
    patch2 = alpha * patch1 + (1 - alpha) * decorrNoise
    # shift second patch
    patch2 = patchShift(patch2, dr, dc, overSample=overSample)
    if bandwidth < 1.:
        limitBandwidth(patch1, bandwidth)
        limitBandwidth(patch2, bandwidth)
    return patch1, patch2


def nccIntPatches(patchS, patchR, ampMatch=False):
    '''
    Cross correlate patches using circular convolution via ffts with single
    pixel accuracy
    Parameters
    ----------
    patchS : 2d complex data
        Search patch.
    patchR : 2d complex data
        shifted patch.
    ampMatch : bool, optional
        Peform amplitude rather than complex matching. The default is False.
    Returns
    -------
    ncc : 2d floating point
        Normalized cross correlation.

    '''
    if ampMatch:
        ampS = np.abs(patchS)
        meanS = np.mean(ampS)
        sigmaS = np.std(ampS)
        ampR = np.abs(patchR)
        meanR = np.mean(ampR)
        sigmaR = np.std(ampR)
        fftForwardS = np.fft.fft2(ampS - meanS)
        fftForwardR = np.fft.fft2(ampR - meanR)
    else:
        fftForwardS = np.fft.fft2(patchS)
        fftForwardR = np.fft.fft2(patchR)
        sigmaS = np.std(patchS)
        sigmaR = np.std(patchR)
    # Compute cross corr from spectra
    nr, nc = patchS.shape
    # spectrum
    ccSpectrum = fftForwardS * np.conjugate(fftForwardR)
    #
    ncc = np.fft.ifft2(ccSpectrum)/(sigmaS * sigmaR * nr * nc)
    #
    ncc = np.fft.fftshift(ncc)
    #
    return ncc


def _fitfunc(p, xp, yp):
    return p[4] * np.exp(-(xp - p[0])**2 / p[2]**2 - (yp - p[1])**2 / p[3]**2)


def _errfunc(p, xp, yp, ccPeak):
    return _fitfunc(p, xp, yp) - ccPeak


def gaussFit(ccPeak, overSampleFactor, debug=False):
    '''
    Fit a gaussian to the cross correlation peak.

    Parameters
    ----------
    ccPeak : np array
        Area around correlation peak.
    overSampleFactor : int
        Oversample factor (1 or 2).
    debug : bool, optional
        Add the fitted retult to the return. The default is False.

    Returns
    -------
    rMaxOs: float
        Oversampled row coordinate.
    cMaxOs: float
        Oversampled column coordinate..
    rhow: float
        Correlation max as determined by fitted peak.
    '''
    rhw, chw = ccPeak.shape[0]/2, ccPeak.shape[1]/2
    cp, rp = np.meshgrid(np.linspace(-chw, rhw, ccPeak.shape[1]),
                         np.linspace(-rhw, rhw, ccPeak.shape[0]))
    p0 = [0, 0, 25, 25, 0.5]
    p1, success = optimize.leastsq(_errfunc, p0[:],
                                   args=(rp.flatten(), cp.flatten(),
                                         ccPeak.flatten()))
    # oversampled offsets and corr peak from fit
    rMaxOs, cMaxOs, rhow = p1[0] + rhw, p1[1] + rhw, p1[4]
    if not debug:
        return rMaxOs, cMaxOs, rhow
    # return gaussian for debugging/illustration purposes
    return rMaxOs, cMaxOs, rhow, _fitfunc(p1, rp, cp), p1


def osSubPixGaussian(cc, overSampleCorr, overSamplePeak, boxS, debug=False):
    '''
    Use fit to to Gaussian peak to oversample.
    Parameters
    ----------
     cc : 2d array
        Integer sampled or 2x oversampled correlation function.
    overSampleCorr : int
        OverSample factor for initial cross corr (1x or 2x)
    overSamplePeak : int
        Oversample factor (not including any 2x oversample).
    boxSize : int
        Size ("radius") of box in non-oversampled pixels around peak to
        oversample +/- box size.
    Returns
    -------
    dr : float
        row sub-pixel offset.
    dc : float
        column subpixel value.
    cmax : float
        max correlation value.
    ccPeak : 2d np float array
        Estimated peak.
    '''
    boxUse = boxS * overSampleCorr
    # location of peak in patch
    rMax, cMax = np.unravel_index(np.argmax(np.abs(cc)),  cc.shape)
    # get area around peak
    ccPeak = cc[rMax - boxUse: rMax + boxUse, cMax - boxUse: cMax + boxUse]
    #
    ccPeakOs = overSamplePatch(ccPeak, overSample=overSamplePeak)
    # find peak for subpixel offsets
    returnValues = gaussFit(np.real(ccPeakOs), overSamplePeak, debug=debug)
    rMaxOs, cMaxOs, rhow = returnValues[0:3]
    # Oversampled offsets
    rMax1 = rMax * overSamplePeak + (rMaxOs - boxUse * overSamplePeak)
    cMax1 = cMax * overSamplePeak + (cMaxOs - boxUse * overSamplePeak)
    # Compute final offsets
    rMaxF = ((rMax1/overSamplePeak) - cc.shape[0] / 2.) / overSampleCorr
    cMaxF = ((cMax1/overSamplePeak) - cc.shape[1] / 2.) / overSampleCorr
    # (rMax / overSampleFactor - boxS) * 1. / patchOver
    if not debug:
        return rMaxF, cMaxF, np.max(np.abs(cc)), ccPeakOs
    return rMaxF, cMaxF, np.max(np.abs(cc)), ccPeakOs, returnValues[3]


def osSubPix(cc, overSampleCorr, overSamplePeak, boxS):
    '''
    Use brute force oversampling to determine the max around the correlation
    peak.
    Parameters
    ----------
    cc : 2d array
        Integer sampled or 2x oversampled correlation function.
    overSampleCorr : int
        OverSample factor for initial cross corr (1x or 2x)
    overSamplePeak : int
        Oversample factor (not including any 2x oversample).
    boxSize : int
        Size ("radius") of box in non-oversampled pixels around peak to
        oversample +/- box size.
    Returns
    -------
    dr : float
        row sub-pixel offset.
    dc : float
        column subpixel value.
    cmax : float
        max correlation value.
    ccPeak : 2d np float array
        Oversampled peak (or similar from other subpix matcher).
    '''
    # expand box size to account for an overSample
    boxUse = boxS * overSampleCorr
    # location of peak in patch
    rMax, cMax = np.unravel_index(np.argmax(np.abs(cc)),  cc.shape)
    # get area around peak
    ccPeak = cc[rMax - boxUse: rMax + boxUse, cMax - boxUse: cMax + boxUse]
    # oversample peak
    ccPeakOs = overSamplePatch(ccPeak, overSample=overSamplePeak)
    # find peak for subpixel offsets
    rMaxOs, cMaxOs = np.unravel_index(np.argmax(ccPeakOs),  ccPeakOs.shape)
    # Oversampled offsets
    rMax1 = rMax * overSamplePeak + (rMaxOs - boxUse * overSamplePeak)
    cMax1 = cMax * overSamplePeak + (cMaxOs - boxUse * overSamplePeak)
    # Compute final offsets
    rMaxF = ((rMax1/overSamplePeak) - cc.shape[0] / 2.) / overSampleCorr
    cMaxF = ((cMax1/overSamplePeak) - cc.shape[1] / 2.) / overSampleCorr
    # (rMax / overSampleFactor - boxS) * 1. / patchOver
    return rMaxF, cMaxF, np.max(np.abs(cc)), ccPeakOs


def nccPatches(patchS, patchR, ampMatch=False, overSampleCorr=True,
               subPix=osSubPix, subPixArgs=(10, 4), subPixKwArgs={}):
    '''
    Correlate patches to determine sub-pixel offsets.
    Parameters
    ----------
    patchS : 2d complex data
        Search patch.
    patchR : 2d complex data
        shifted patch.
    ampMatch : bool, optional
        Peform amplitude rather than complex matching. The default is False.
    overSampleCorr : bool, optional
        Apply 2x oversampling before matching. The default is True.
    subPix : func, optional
        The function used for sub-pixel matching. The default is osSubPix.
    subPixArgs : TYPE, optional
        args for subpixel match. The default is [10].
    subPixKwArgs : TYPE, optional
        kwargs for subpixel matcher. The default is None.
    Returns
    -------
    dr : float
        row sub-pixel offset.
    dc : float
        column subpixel value.
    cmax : float
        max correlation value.
    cc : 2d np float array
        Integer or oversampled cross correlation grid.
    ccPeak : 2d np float array
        Oversampled peak (or similar from other subpix matcher).
    '''
    # Do basic cross correlation
    if overSampleCorr:
        patchSx2 = overSamplePatch(patchS, overSample=2)
        patchRx2 = overSamplePatch(patchR, overSample=2)
        cc = nccIntPatches(patchSx2, patchRx2, ampMatch=ampMatch)
        osFactor = 2
        patchShape = patchSx2.shape
    else:
        cc = nccIntPatches(patchS, patchR, ampMatch=ampMatch)
        patchShape = patchS.shape
        osFactor = 1
    #
    drInt, dcInt = np.unravel_index(np.argmax(np.abs(cc)), patchShape)
    erInt = np.abs(drInt - patchShape[0]/2) / osFactor
    ecInt = np.abs(dcInt - patchShape[1]/2) / osFactor
    if erInt > 2 or ecInt > 2:
        return np.nan, np.nan, np.nan, cc, []
    #
    dr, dc, cmax, ccPeak = subPix(cc, osFactor, *subPixArgs, **subPixKwArgs)
    if dr > 2 or dc > 2:
        return np.nan, np.nan, np.nan, cc, []
    return dr, dc, cmax, cc, ccPeak


def overSamplePatch(patch, overSample=7):
    '''
    Oversample a patch of synthetic data
    Parameters
    ----------
    patch : complex data
        Patch with synthetic data.
    overSample : int, optional
        Oversample factor. The default is 7.
    Returns
    -------
    patchOver : 2D complex data
        Oversampled data.
    '''
    #
    osShape = tuple(x * overSample for x in patch.shape)
    fftZeroPad = np.zeros(osShape, dtype=np.complex64)
    # For now only use even oversample
    if patch.shape[0] % 2 != 0 or patch.shape[0] % 2 != 0:
        print(f'use even patch for oversampling {patch.shape}')
        exit()
    #
    r0, c0 = int(osShape[0] / 2), int(osShape[1] / 2)  # patch centers
    rhw, chw = int(patch.shape[0]/2), int(patch.shape[1]/2)  # patch half width
    # fft forward
    fftForward = np.fft.fft2(patch)
    # zero pad to acomplish oversample
    fftZeroPad[r0 - rhw: r0 + rhw, c0 - chw: c0 + chw] = \
        np.fft.fftshift(fftForward)
    # inverse transform
    patchOver = np.fft.ifft2(np.fft.ifftshift(fftZeroPad)) * overSample**2
    return patchOver.astype(np.complex64)


def patchShift(patch, dr, dc, overSample=7):
    '''
    Shift a patch by dr/overSample by dc/overSample
    Parameters
    ----------
    patch : 2d complex
        Patch to shift.
    dr : int
        shift dr/overSample fraction of a pixel.
    dc : int
        shift dc/overSample fraction of a pixel.
    overSample : int, optional
        Oversample factor. The default is 7.
    Returns
    -------
    patchShift : TYPE
        DESCRIPTION.

    '''
    # Ensure offset is less than dr/overSample
    dr, dc = dr % overSample, dc % overSample
    patchOs = overSamplePatch(patch, overSample=overSample)
    patchShift = patchOs[dr::overSample, dc::overSample]
    return patchShift


class speckleJobs:
    '''
    Object to run multiple threads of the same function
    As an example:
        # Setup a function to multithread
        fftJobs = nf.speckleJobs(nf.whiteNoisePatch)
        # run 10 instances of whiteNoisePatch(64,64)
        for ii in range(0,10):
            fftJobs.addQ(64,64)
        # run the 10 instances
        fftJobs.runThreads()
    '''

    def __init__(self, myFunction, maxThreads=multiprocessing.cpu_count()/2):
        '''
        Init speckle Jobs.
        Parameters
        ----------
        myFunction : function
            Function being multi-threaded.
        maxThreads : int, optional
            maxThreads. The default is multiprocessing.cpu_count()/2.
        Returns
        -------
        None.
        '''
        self.myFunction = myFunction
        self.results = []  # List with returned results
        self.q = Queue(maxsize=0)  # Q for jobs processing
        self.nJobs = 0  # Number of jobs
        self.maxThreads = maxThreads
        self.nRunning = 0  # Number of currently running threads

    def runMyFunction(self):
        '''
        Run a job from the queue
        Returns
        -------
        None.
        '''
        while not self.q.empty():
            work = self.q.get()
            try:
                data = self.myFunction(*work[1], **work[2])
                self.results[work[0]] = copy.deepcopy(data)
                if work[0] % (self.nJobs/5) == 0:
                    print('.', end='')
                    sys.stdout.flush()
            except Exception:
                print(data)
                self.results[work[0]] = -1
            self.nRunning -= 1
            self.q.task_done()
            return

    def addQ(self, *args, **kwargs):
        '''
        Add a job to the queue
        Parameters
        ----------
        *args : list
            pass through arguments.
        **kwargs : dict
            pass through keywords.
        Returns
        -------
        None.
        '''
        # add jobs to q
        self.q.put((self.nJobs, args, kwargs))
        self.nJobs += 1  # increment job count
        self.results.append([])  # Add item for output

    def runThreads(self):
        '''
        Run the jobs in the Q
        Returns
        -------
        None.
        '''
        jobsStarted = 0
        while jobsStarted < self.nJobs:
            if self.nRunning < self.maxThreads:
                workerThread = Thread(target=self.runMyFunction, daemon=True)
                self.nRunning += 1
                jobsStarted += 1
                workerThread.start()
            else:
                time.sleep(0.001)
        self.q.join()  # wait for completion
        return

    def runNInstances(self, nJobs, *args, **kwargs):
        '''
        Run the jobs (e.g., random number generator) with same args nJobs times
        Parameters
        ----------
        nJobs : int
            Number of threads to run.
        *args : list
            pass through arguments.
        **kwargs : dict
            pass through keywords.
        Returns
        -------
        None.
        '''
        for ii in range(0, nJobs):
            self.addQ(*args, **kwargs)
        self.runThreads()
        print('x', end='\r')
        sys.stdout.flush()
        return

    def reset(self):
        '''
        Reset so the same function can be run again.
        Returns
        -------
        None.
        '''
        self.results = []  # List with returned results
        self.q = Queue(maxsize=0)  # Q for jobs processing
        self.nJobs = 0  # Number of jobs
