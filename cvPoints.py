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
import sys
from IPython.display import Markdown as md
    
class cvPoints :
    """ cvpoints object - use for cvpoints  data """
    #
    #===================== Cv Setupstuff Stuff ============================ 
    # 
    def __init__(self,cvFile=None ) :
        """ \n\nRead a cvfile and manipulate cv points """
        #
        # set everything to zero as default
        self.cvFile=None
        self.lat,self.lon=np.array([]),np.array([])
        self.x,self.y=np.array([]),np.array([])
        self.z=np.array([])
        self.nocull=None
        self.pound2=False
        self.header=[]
        self.vx,self.vy,self.vz=np.array([]),np.array([]),np.array([])
        #
        self.setCVFile(cvFile)
        self.epsg=None
        self.llproj=pyproj.Proj("+init=EPSG:4326")
        self.xyproj=None
        if cvFile != None :
            self.readCVs(cvFile=cvFile)
        
    def setCVFile(self,cvFile) :
        ''' set the name of the CVlfile '''
        self.cvFile = cvFile            

    def checkCVFile(self) :
        ''' check cvfile exists '''
        if self.cvFile == None :
            myerror("No cvfile specified")
        if not os.path.exists(self.cvFile) :
            myerror("cvFile : {0:s} does not exist".format(self.cvFile))
            
    def setEPSG(self) :
        ''' set epsg based on northern or southern lat '''
        if len(self.lat) <= 0 : 
            myerror("Cannot set epsg without valid latlon")  
        self.epsg=[3031,3413][self.lat[0] > 0]
        self.xyproj=pyproj.Proj(f"+init=EPSG:{self.epsg}")           
    #    
    #===================== Cv input/output stuff ============================ 
    #       
    def readCVs(self,cvFile=None) :
        ''' read cvs, set projection based on hemisphere, convert to x,y (m)'''
        self.setCVFile(cvFile)
        self.checkCVFile()
        # 
        cvFields=[latv,lonv,zv,vxv,vyv,vzv]=[],[],[],[],[],[]
        # loop through file to read points
        fpIn=open(self.cvFile,'r')
        for line in fpIn :
            if '#' in line and '2' in line :
                self.pound2=True
            if '&' not in line and ';' not in line and '#' not in line :
                #lat,lon,z,vx,vy,vz =
                for x,y in zip(cvFields,[float(x) for x in line.split()[0:6]]) :
                    x.append(y)       
        fpIn.close()
        # append to any existing points
        for x,y in zip(['lat','lon','z','vx','vy','vz'],cvFields) :
            setattr(self,f'{x}',np.append(getattr(self,f'{x}'),y))
        #
        self.vh=np.sqrt(self.vx**2 + self.vy**2)          
        # 
        self.setEPSG()
        self.nocull=np.ones(self.vx.shape,dtype=bool) # set all to no cull
        self.x,self.y=self.lltoxym(self.lat,self.lon) # convert to xy coords
            
    def writeCVs(self,cvFileOut) :
        ''' write cvs '''
        fpOut=open(cvFileOut,'w')
        # in case type file that needs this.
        for line in self.header :
            print(f'; {line}',file=fpOut)
        if self.pound2 : 
            print('# 2',file=fpOut)
        #
        for lat,lon,z,vx,vy,vz,nocull in zip(self.lat,self.lon,self.z,self.vx,self.vy,self.vz,self.nocull) :
            if nocull : # only print non-culled points
                print(f'{lat:10.5f} {lon:10.5f} {z:8.1f} {vx:8.1f} {vy:8.1f} {vz:8.1f}',file=fpOut )
        print('&',file=fpOut)
        fpOut.close()
    #    
    #===================== CV Select Stuff ============================ 
    #
    def zeroCVs(self) :
        '''return zero cvpoints locations'''
        return np.abs(self.vh) < 0.00001
     
    def allCVs(self) :
        ''' return allCvpoint locations'''
        return np.abs(self.vh) >= 0
     
    def NallCVs(self) :
        ''' return number of cvpoints '''
        return sum(self.allCVs()) 
    
    def vRangeCVs(self,minv,maxv) :
        ''' cvs in range (minv,maxv) '''
        return np.logical_and(self.vh >= minv,self.vh <= maxv )
    
    def NzeroCVs(self) :
        ''' return number of zero cvpoints'''
        return sum(self.zeroCVs())  
     
    def NVRangeCVs(self,minv,maxv) :
        ''' number of cv points in range (minv,maxv) '''
        return sum(self.vRangeCVs(minv,maxv))
    
    def NvAllCVs(self,minv,maxv) :
        return sum(self.allCVs(minv,maxv)) 
    #    
    #===================== Coordinate Stuff ============================ 
    #
    def lltoxym(self,lat,lon):
        ''' convert ll to xy '''
        if self.xyproj != None :
            x,y=pyproj.transform(self.llproj,self.xyproj,lon,lat)
            return x,y
        else :
            myerror("lltoxy : proj not defined")    
            
    def llzero(self) :
        ''' return lat.lon of zero cvs'''
        iZero=self.zeroCVs()
        return self.lat(iZero),self.lat(iZero)
        
    def xyzerom(self) :
        ''' xy coordinates in m for zero points'''
        iZero=self.zeroCVs()
        return self.x[iZero],self.y[iZero]
    
    def xyzerokm(self) :
        ''' xy coordinates in km for zero points'''
        x,y=self.xyzerom()
        return x/1000.,y/1000.
        
    def xyallm(self) :
        ''' xy coordinates in m for all points'''
        return self.x,self.y
   
    def xyallkm(self) :
        ''' xy coordinates in km for all points'''
        return self.x/1000.,self.y/1000.
        
    def xyVRangem(self,minv,maxv) :
        ''' xy coordinates in m for points with speed in range (minv,maxv)'''
        iRange=self.vRangeCVs(minv,maxv)
        return self.x[iRange],self.y[iRange]
        
    def xyVRangekm(self,minv,maxv) :
        ''' xy coordinates in km for points with speed in range (minv,maxv)'''
        x,y=self.xyvRangem(minv,maxv)
        return x/1000.,y/1000.
    #
    #===================== Interpolate Velocity Stuff ============================ 
    #  
    def _cvVels(func) :
        ''' decorator to interpolate cv values from vel map '''
        @functools.wraps(func)
        def cvV(*args,**kwargs) :
            x,y=func(*args)
            vx,vy,vr=args[1].interpGeo(x,y,**kwargs)
            iGood=np.isfinite(vx)
            return vx,vy,vr,iGood
        return cvV
    
    @_cvVels
    def zeroVData(self,vel) :
        ''' get velocicvs from vel map for zero points'''
        return self.xyzerom()
        
    @_cvVels
    def allVData(self,vel) :
        ''' get veloccvs from vel map for all points'''
        return self.xyallm()  
    
    @_cvVels
    def vRangeData(self,vel,minv,maxv) :
        ''' get velocicvs from vel map for points in range (vmin,vmax)'''
        return self.xyVRangem(minv,maxv) 
    #
    #===================== Stats Stuff ============================ 
    #       
    
        
    def _stats(func) :
        ''' decorator for computing stats routines '''
        
        @functools.wraps(func)
        def mstd(*args,table=None,**kwargs) :
            x,y,iPts=func(*args)
            vx,vy,vr=args[1].interpGeo(x,y,**kwargs)
            # subtract cvpoint values args[0] is self
            dx,dy=vx-args[0].vx[iPts],vy-args[0].vy[iPts]
            iGood=np.isfinite(vx)
            def statsTable(mux,muy,sigx,sigy,nPts) :
                ''' make a markdown table '''
                return mux,muy,sigx,sigy,nPts,md(f'|Statistic | $u_x-v_x$ (m/yr)|$u_y-v_y$ (m/yr)| N points|\n' \
                   f'|-----|---------|---------|----|\n' \
                   f'|Mean| {mux:0.2}|{muy:0.2}| {nPts}|\n' \
                   f'|Std.Dev.| {sigx:0.2}|{sigy:0.2}| {nPts}|')
            if table == None :
                return np.average(dx[iGood]),np.average(dy[iGood]),np.std(dx[iGood]),np.std(dy[iGood]),sum(iGood)
            else :
                return statsTable(np.average(dx[iGood]),np.average(dy[iGood]),np.std(dx[iGood]),np.std(dy[iGood]),sum(iGood))
        return mstd
    
        
     
    @_stats
    def zeroStats(self,vel) :
        ''' pass in vel and get stats of zero points '''
        x,y=self.xyzerom()
        iPts=self.zeroCVs()
        return x,y,iPts
        
    @_stats
    def allStats(self,vel) :
        ''' return stats for all points '''
        x,y=self.xyallm()
        iPts=self.allCvs()
        return x,y,iPts
    
    @_stats
    def vRangeStats(self,vel,minv,maxv) :
        ''' get stats for cvpoints in range (minv,maxv) '''
        x,y=self.xyVRangem(minv,maxv)
        iPts=self.vRangeCVs(minv,maxv)
        return x,y,iPts
    
    #
    #===================== Plot Cv locations Stuff ============================ 
    #    
    def _plotCVLocs(func) :
      @functools.wraps(func)
      def plotCVXY(*args,vel=None,ax=None,**kwargs) :
          x,y=func(*args)
          #xm,ym=self.xyallm()
          if vel != None :
              vx,vy,vr=vel.interpGeo(x,y)
              iGood=np.isfinite(vx)
              x=x[iGood]
              y=y[iGood]
          if ax==None :
              plt.plot(x*0.001,y*0.001,'.',**kwargs)
          else :
              ax.plot(x*0.001,y*0.001,'.',**kwargs)
      return plotCVXY
                 
    @_plotCVLocs
    def plotAllCVLocs(self) :
        ''' plot x,y locations of all points '''
        return self.xyallm()
    
    @_plotCVLocs
    def plotZeroCVLocs(self) :
        ''' plot x,y locations of zero points '''
        return self.xyzerom()  
    
    @_plotCVLocs
    def plotVRangeCVLocs(self,minv,maxv) :
        ''' plot x,y locations of zero points '''
        return self.xyVRangem(minv,maxv)  
    #
    #===================== CV culling stuff  Stuff ============================ 
    #   
    def readCullFile(self,cullFile) :
        ''' read a cull file '''
        fp=open(cullFile,'r')
        myParams=eval(fp.readline())
        print(myParams["cvFile"])
        print(self.cvFile)
        cullPoints=[]
        for line in fp :
            cullPoints.append(int(line))
        if len(cullPoints) != myParams["nBad"] :
            myerror(f'reading culled points expected {myParams["nBad"]} but only found {len(cullPoints) }')
        #
        fp.close()
        return np.array(cullPoints)  
        
    def applyCullFile(self,cullFile )  :
        ''' read a cvpoint cull file and update nocull'''
        #
        self.header.append(cullFile)
        toCull=self.readCullFile(cullFile)
        if len(toCull) > 0 :
            self.nocull[toCull]=False
    
#
#===================== Other Stuff outside class deff============================ 
#   
def myerror(message,myLogger=None):
    """ print error and exit """
    print('\n\t\033[1;31m *** ',message,' *** \033[0m\n') 
    if myLogger != None :
        myLogger.logError(message)
    sys.exit()