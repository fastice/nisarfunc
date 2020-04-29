# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:35:53 2019

@author: ian
"""

import os
import numpy as np
import pyproj
import functools
import matplotlib.pylab as plt
from nisarfunc import myError
from IPython.display import Markdown as md


class cvPoints:
    ''' cvpoints object - Has functionality for:
        input/output of cal/val points,
        filtering of points based on speed,
        computing the differences with a veloity map
        computing the statistics of these differences, and
        plotting the differencs and loctions'''

    labelFontSize = 16  # Font size for plot labels
    plotFontSize = 15  # Font size for plots
    legendFontSize = 15  # Font size for legends

    def __init__(self, cvFile=None, epsg=None):
        """ \n\nRead a cvfile and manipulate cv points """
        #
        # set everything to zero as default
        self.cvFile = None  # File with cal/val points
        self.lat, self.lon = np.array([]), np.array([])  # lat/lon of points
        self.x, self.y = np.array([]), np.array([])  # x/y PS of points in m
        self.z = np.array([])  # Elevatin of points
        self.nocull = None  # Used to mark points to cull (by default all 1)
        self.pound2 = False  # Flag to write pound 2 to output (historical)
        self.header = []  # In dictionary used as header - needs some work
        self.vx, self.vy, self.vz = np.array([]), np.array([]), np.array([])
        #
        self.setCVFile(cvFile)
        self.epsg = epsg  # Epsg set 3413 or 3031 depending on if N or S
        self.llproj = pyproj.Proj("EPSG:4326")  # Used for conversions
        self.xyproj = None  # Defined once epsg set.
        if cvFile is not None:
            self.readCVs(cvFile=cvFile)

    def setCVFile(self, cvFile):
        ''' Set the name of the CVlfile '''
        self.cvFile = cvFile

    def checkCVFile(self):
        ''' Check cvfile exists. '''
        if self.cvFile is None:
            myError("No cvfile specified")
        if not os.path.exists(self.cvFile):
            myError("cvFile: {0:s} does not exist".format(self.cvFile))

    def setEPSG(self):
        ''' Set epsg based on northern or southern lat. '''
        if len(self.lat) <= 0:
            myError("Cannot set epsg without valid latlon")
        self.epsg = [3031, 3413][self.lat[0] > 0]
        self.xyproj = pyproj.Proj(f"EPSG:{self.epsg}")
    #
    # ===================== Cv input/output stuff ============================
    #

    def readCVs(self, cvFile=None):
        ''' read cvs, set projection based on hemisphere, convert to x,y (m)'''
        self.setCVFile(cvFile)
        self.checkCVFile()
        #
        cvCols = [latv, lonv, zv, vxv, vyv, vzv] = [], [], [], [], [], []
        # loop through file to read points
        fpIn = open(self.cvFile, 'r')
        for line in fpIn:
            if '#' in line and '2' in line:
                self.pound2 = True
            if '&' not in line and ';' not in line and '#' not in line:
                lineFields = [float(x) for x in line.split()[0:6]]
                for x, y in zip(cvCols, lineFields):
                    x.append(y)
        fpIn.close()
        # append to any existing points
        for var, cvCol in zip(['lat', 'lon', 'z', 'vx', 'vy', 'vz'], cvCols):
            setattr(self, f'{var}', np.append(getattr(self, f'{var}'), cvCol))
        #
        self.vh = np.sqrt(self.vx**2 + self.vy**2)
        #
        self.setEPSG()
        self.nocull = np.ones(self.vx.shape, dtype=bool)  # set all to no cull
        self.x, self.y = self.lltoxym(self.lat, self.lon)  # to xy coords

    def writeCVs(self, cvFileOut):
        ''' Write a CV file as a list of points lat,lon,z,vx,vy,vz with a
        dict header. Need to update to add header. '''
        fpOut = open(cvFileOut, 'w')
        # in case type file that needs this.
        for line in self.header:
            print(f'; {line}', file=fpOut)
        if self.pound2:
            print('# 2', file=fpOut)
        #
        myCVFields = zip(
            self.lat, self.lon, self.z, self.vx, self.vy, self.vz, self.nocull)
        for lat, lon, z, vx, vy, vz, nocull in myCVFields:
            if nocull:  # only print non-culled points
                print(f'{lat:10.5f} {lon:10.5f} {z:8.1f}'
                      ' {vx:8.1f} {vy:8.1f} {vz:8.1f}', file=fpOut)
        print('&', file=fpOut)  # End of data for historical reasons
        fpOut.close()
    #
    # ===================== CV Select Stuff ============================
    #

    def zeroCVs(self):
        '''Return zero cvpoints locations.'''
        return np.abs(self.vh) < 0.00001

    def vRangeCVs(self, minv, maxv):
        ''' CVs in range (minv,maxv). '''
        return np.logical_and(self.vh >= minv, self.vh < maxv)

    def allCVs(self):
        ''' Return allCvpoint locations.'''
        return np.abs(self.vh) >= 0

    def NzeroCVs(self):
        ''' return number of zero cvpoints'''
        return sum(self.zeroCVs())

    def NVRangeCVs(self, minv, maxv):
        ''' number of cv points in range (minv,maxv) '''
        return sum(self.vRangeCVs(minv, maxv))

    def NallCVs(self):
        ''' Return number of cvpoints. '''
        return sum(self.allCVs())
    #
    # ===================== Interpolate Velocity Stuff ====================
    #

    def _cvVels(func):
        ''' decorator to interpolate cv values from vel map '''
        @functools.wraps(func)
        def cvV(*args, **kwargs):
            x, y = func(*args)
            vx, vy, vr = args[1].interp(x, y, ['vx', 'vy', 'vz'], **kwargs)
            iGood = np.isfinite(vx)
            return vx, vy, vr, iGood
        return cvV

    @_cvVels
    def vRangeData(self, vel, minv, maxv):
        ''' Get velocity from vel map for points in range (vmin,vmax).'''
        return self.xyVRangem(minv, maxv)

    #
    # ===================== Stats Stuff ============================
    #

    def _stats(func):
        ''' decorator for computing stats routines '''
        # create a table
        def statsTable(muX, muY, sigX, sigY, rmsX, rmsY, nPts):
            ''' make a markdown table '''
            myTable = md(f'|Statistic|$u_x - v_x$ (m/yr)|$u_y - v_y$ (m/yr)|'
                         f'N points|\n'
                         f'|------|------------|------------|---------|\n'
                         f'|Mean|{muX:0.2}|{muY:0.2}|{nPts}|\n'
                         f'|Std.Dev.|{sigX:0.2}|{sigY:0.2}|{nPts}|\n'
                         f'|rms|{rmsX:0.2}|{rmsY:0.2}|{nPts}|')
            return muX, muY, sigX, sigY, rmsX, rmsY, nPts, myTable

        # the actual stats
        @functools.wraps(func)
        def mstd(*args, table=False, **kwargs):
            x, y, iPts = func(*args)
            dvx, dvy, iGood = args[0].cvDifferences(x, y, iPts, args[1])
            muX, muY = np.average(dvx), np.average(dvy)
            rmsX = np.sqrt(np.average(dvx**2))
            rmsY = np.sqrt(np.average(dvy**2))
            sigX, sigY = np.std(dvx), np.std(dvy)
            if not table:
                return muX, muY, sigX, sigY, rmsX, rmsY, sum(iGood)
            else:
                return statsTable(muX, muY, sigX, sigY, rmsX, rmsY, sum(iGood))
        return mstd

    @_stats
    def vRangeStats(self, vel, minv, maxv):
        ''' get stats for cvpoints in range (minv,maxv) '''
        x, y = self.xyVRangem(minv, maxv)
        iPts = self.vRangeCVs(minv, maxv)
        return x, y, iPts
    #
    # ===================== Coordinate Stuff ============================
    #

    def lltoxym(self, lat, lon):
        ''' Convert ll to xy. '''
        if self.xyproj is not None:
            x, y = pyproj.transform(self.llproj, self.xyproj, lat, lon)
            return x, y
        else:
            myError("lltoxy: proj not defined")

    def llzero(self):
        ''' Return lat and lon of zero CVs. '''
        iZero = self.zeroCVs()
        return self.lat(iZero), self.lat(iZero)

    def xyzerom(self):
        ''' XY coordinates in m for zero points.'''
        iZero = self.zeroCVs()
        return self.x[iZero], self.y[iZero]

    def xyzerokm(self):
        ''' XY coordinates in km for zero points.'''
        x, y = self.xyzerom()
        return x/1000., y/1000.

    def xyallm(self):
        ''' xy coordinates in m for all points'''
        return self.x, self.y

    def xyallkm(self):
        ''' xy coordinates in km for all points'''
        return self.x/1000., self.y/1000.

    def xyVRangem(self, minv, maxv):
        ''' xy coordinates in m for points with speed in range (minv,maxv)'''
        iRange = self.vRangeCVs(minv, maxv)
        return self.x[iRange], self.y[iRange]

    def xyVRangekm(self, minv, maxv):
        ''' xy coordinates in km for points with speed in range (minv,maxv)'''
        x, y = self.xyvRangem(minv, maxv)
        return x/1000., y/1000.
    #
    # ===================== Plot Cv locations Stuff ========================
    #

    def _plotCVLocs(func):
        ''' Decorator for plotting locations in range (vmin,vmax). '''
        @functools.wraps(func)
        def plotCVXY(*args, vel=None, ax=None, nSig=3, **kwargs):
            x, y = func(*args)
            #
            if vel is not None:
                vx, vy, vr = vel.interp(x, y, ['vx', 'vy'])
                iGood = np.isfinite(vx)
                x, y = x[iGood], y[iGood]
            if ax is None:
                plt.plot(x*0.001, y*0.001, '.', **kwargs)
            else:
                ax.plot(x*0.001, y*0.001, '.', **kwargs)
        return plotCVXY

    @_plotCVLocs
    def plotVRangeCVLocs(self, minv, maxv):
        ''' plot x,y locations for points where maxv > v > min '''
        return self.xyVRangem(minv, maxv)

    @_plotCVLocs
    def plotOutlierLocs(self, minv, maxv, vel, nSig=3):
        ''' Plot outliers where difference in either velocity component is >
        nSig*sigma '''
        x, y = self.xyVRangem(minv, maxv)  # get points in range
        iPts = self.vRangeCVs(minv, maxv)  # Just points in range (minv,vmaxv)
        dvx, dvy, iGood = self.cvDifferences(x, y, iPts, vel)  # get diffs
        x, y = x[iGood], y[iGood]  # points in range with valid diffs
        sigX, sigY = np.std(dvx), np.std(dvy)  # Sigmas
        # Find outliers
        iOut = np.logical_or(np.abs(dvx) > nSig * sigX,
                             np.abs(dvy) > nSig * sigY)
        return x[iOut], y[iOut]

    #
    # ===================== CV plot differences ===============================
    #

    def showDiffs(self, vel, minv, maxv):
        ''' Plot differences and histograms of differences '''
        figD = plt.figure(figsize=(14, 5))
        self.plotVRangeCVDiffs(vel, figD, minv, maxv)
        self.plotVRangeHistDiffs(vel, figD, minv, maxv)
        plt.tight_layout()

    def _plotDiffs(func):
        ''' Decorator for plotting differences between tie points and
        velocities interpolated from in vel map '''
        @functools.wraps(func)
        def plotp(*args, **kwargs):
            x, y, iPts = func(*args)
            dx, dy, iGood = args[0].cvDifferences(x, y, iPts, args[1])
            axImage = args[2].add_subplot(131)
            axImage.plot(dx, 'r.')
            axImage.plot(dy, 'b.')
            axImage.set_xlabel('Point', size=args[0].labelFontSize)
            axImage.set_ylabel('$u_x-v_x, u_y-v_y$ (m/yr)',
                               size=args[0].labelFontSize)
            axImage.legend(['$u_x-v_x$', '$u_y-v_y$'],
                           fontsize=args[0].legendFontSize)
            axImage.tick_params(axis='x', labelsize=args[0].plotFontSize)
            axImage.tick_params(axis='y', labelsize=args[0].plotFontSize)
            return axImage
        return plotp

    def _histDiffs(func):
        ''' Decorator for plotting differences between tie points and
        velocities interpolated from in vel map '''
        # the actual stats
        def clipTail(dx):
            ''' Clip differences to +/- 3 sigma '''
            threeSigma = 3 * np.std(dx)
            dx[dx < -threeSigma] = -threeSigma
            dx[dx > threeSigma] = threeSigma
            return dx

        @functools.wraps(func)
        def ploth(*args, **kwargs):
            x, y, iPts = func(*args)
            dx, dy, iGood = args[0].cvDifferences(x, y, iPts, args[1])
            dx, dy = clipTail(dx), clipTail(dy)  # Set vals > 3sig to 3sig
            axHistX = args[2].add_subplot(132)
            axHistY = args[2].add_subplot(133)
            colors = {'x': 'r', 'y': 'b'}
            # produce and label plots for x and y compnents
            for var, lab, axH in zip([dx, dy], ['x', 'y'], [axHistX, axHistY]):
                axH.hist(var, bins='auto', density=True, color=colors[lab])
                axH.set_xlabel(f'$u_{lab}-v_{lab}$ (m/yr)',
                               size=args[0].labelFontSize)
                axH.set_ylabel('Probability', size=args[0].labelFontSize)
                axH.tick_params(axis='x', labelsize=args[0].plotFontSize)
                axH.tick_params(axis='y', labelsize=args[0].plotFontSize)
            return axHistX, axHistY
        return ploth

    @_plotDiffs
    def plotVRangeCVDiffs(self, vel, figD, minv, maxv):
        ''' Plot differences between c/v points and interpolated values from
        v in range (minv,maxv).
        Originall written with decorator to accomodate multiple cases,
        but collapsed it down to one. Kept decorator for future mods. '''
        x, y = self.xyVRangem(minv, maxv)
        iPts = self.vRangeCVs(minv, maxv)
        return x, y, iPts

    @_histDiffs
    def plotVRangeHistDiffs(self, vel, figD, minv, maxv):
        ''' Plot differences between c/v points and interpolated values from
        v in range (minv,maxv).
        Originall written with decorator to accomodate multiple cases,
        but collapsed it down to one. Kept decorator for future mods. '''
        x, y = self.xyVRangem(minv, maxv)
        iPts = self.vRangeCVs(minv, maxv)
        return x, y, iPts

    def cvDifferences(self, x, y, iPts, vel):
        ''' Interpolate the velocity components from velocity map at the c/v
        point locations and return the differences (vx_map - vx_cv).'''
        vx, vy, vr = vel.interp(x, y, ['vx', 'vy', 'vz'])
        # subtract cvpoint values args[0] is self
        dvx, dvy = vx - self.vx[iPts], vy - self.vy[iPts]
        iGood = np.isfinite(vx)
        # Return valid points and locations where good
        return dvx[iGood], dvy[iGood], iGood

    #
    # ===================== CV culling stuff  ==============================
    #

    def readCullFile(self, cullFile):
        ''' Read a cull file. For an original cvFile, a culling process was
        applied and the results saved in the cullFile. The file contains a
        header dictionary. Subsequent lines are indices to cull.'''
        fp = open(cullFile, 'r')
        myParams = eval(fp.readline())
        print(myParams["cvFile"])
        print(self.cvFile)
        cullPoints = []
        for line in fp:
            cullPoints.append(int(line))
        if len(cullPoints) != myParams["nBad"]:
            myError(f'reading culled points expected {myParams["nBad"]}'
                    ' but only found {len(cullPoints) }')
        #
        fp.close()
        return np.array(cullPoints)

    def applyCullFile(self, cullFile):
        ''' Read a cv point cull file and update nocull so that these points
        can be filtered out if needed. '''
        #
        self.header.append(cullFile)
        toCull = self.readCullFile(cullFile)
        if len(toCull) > 0:
            self.nocull[toCull] = False
