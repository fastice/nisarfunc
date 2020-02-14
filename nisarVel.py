#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:08:15 2020

@author: ian
"""

# geoimage.py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from utilities.readImage import readImage
from utilities.myerror import myerror
#from utilities import geodat
import os
#import urllib.request
#from osgeo.gdalconst import *
from osgeo import gdal,  osr
from datetime import datetime
import functools
import xarray as xr
import matplotlib.pylab as plt
#----------------------------------------------------------------------------------------------------------------------------------------------------
# class defintion for an image object, which covers PS data described by a geodat
#----------------------------------------------------------------------------------------------------------------------------------------------------
class nisarVel :
    """ nisar velocity map """
    def __init__(self,vx=None,vy=None,v=None,ex=None,ey=None,e=None,sx=None,sy=None,x0=None,y0=None,dx=None,dy=None,verbose=True) :
        ''' initialize a nisar velocity object'''
        self.sx,self.sy,self.x0,self.y0,self.dx,self.dy=sx,sy,x0,y0,dx,dy
        self.xx,self.yy=[],[]
        self.xGrid,self.yGrid=[],[]
        self.useXR=False
        self.vx,self.vy,self.vv,self.ex,self.ey=vx,vy,v,ex,ey
        self.verbose=verbose
        

    #--------------------------------------------------------------------------
    # def setup xy limits
    #--------------------------------------------------------------------------
    def xyCoordinates(self) :
        """ xyCoordinates - setup xy coordinates in km """
        # check specs exist
        if None in (self.x0, self.y0, self.sx, self.sy, self.dx,self.dy) :
            myerror(f'nisarVel.xyCoordinates: x0,y0,sx,sy,dx,dy undefined '\
                     ' {self.x0},{self.y0},s{elf.sx},{self.sy},{self.dx},{self.dy}')
        # remember arange will not generate value for sx*dx (its doing sx-1)
        self.xx=np.arange(self.x0,self.x0+self.sx*self.dx,self.dx)
        self.yy=np.arange(self.y0,self.y0+self.sy*self.dy,self.dy)
        # force the right length
        self.xx,self.yy=self.xx[0:self.sx],self.yy[0:self.sy]   
        

    #--------------------------------------------------------------------------
    # Compute matrix of xy grid points
    #--------------------------------------------------------------------------    
    def xyGrid(self) :
        #
        # if one done grid points not computed, then compute
        if len(self.xx) == 0 :
            self.xyCoordinates()
        #
        # setup array
        self.xGrid,self.yGrid=np.zeros((self.y,self.sx)),np.zeros((self.sy,self.sx))
        for i in range(0,self.sy) :
            self.xGrid[i,:]=self.xx
        for i in range(0,self.sx) :
            self.yGrid[:,i]=self.yy  
            
    def parseMyMeta(self,metaFile) :
        ''' get get from meta file '''
        print(metaFile)
        fp=open(metaFile)
        dates=[]
        for line in fp :
            if 'MM:DD:YYYY' in line :
                tmp=line.split('=')[-1].strip()
                dates.append(datetime.strptime(tmp,"%b:%d:%Y"))
                if len(dates) == 2 :
                    break
        if len(dates) != 2 :
            return None
        fp.close()
        return dates[0]+(dates[1]-dates[0])*0.5
  
    def parseVelCentralDate(self) :

        if(self.fileName != None) :
            metaFile=self.fileName + '.meta'
            if not os.path.exists(metaFile) :
                return None
            return self.parseMyMeta(metaFile)
        return None
        

    def myVariables(self,useVelocity,useErrors,noSpeed=False) :
        ''' select variables '''
        myVars=[]
        if useVelocity :
            myVars+=['vx','vy']
        if not noSpeed :
            myVars+=['vv']
        if useErrors :
            myVars+=['ex','ey']
        return myVars
                   
    #--------------------------------------------------------------------------
    #  setup interpolation functions
    #--------------------------------------------------------------------------
    def setupInterp(self,useVelocity=True,useErrors=False) :
        """ set up interpolation for scalar (xInterp) or velocity/eror (vxInterp,vyInterp,vInterp)  """ 
        if len(self.xx) <= 0 :
            self.xyCoordinates()
        # setup interp - flip xy for row colum
        myVars=self.myVariables(useVelocity,useErrors)
        #
        if self.useXR :
            return
        #
        for myVar in myVars :
            myV=getattr(self, myVar)       
            setattr(self,f'{myVar}Interp', RegularGridInterpolator((self.yy, self.xx), myV, method="linear"))
            

    def __setKey(self, myKey, defaultValue, **kwargs) :
        ''' unpack a keyword from **kwargs '''
        if myKey in kwargs.keys():
            return kwargs(myKey)
        return defaultValue
    
    #--------------------------------------------------------------------------
    # interpolate geo image
    #--------------------------------------------------------------------------
    def interpGeo(self, x, y, **kwargs) :
        ''' Call appropriate interpolation method '''
        if self.useXR :
            return self.interpXR(x,y,**kwargs)
        else :
            return self.interpNP(x,y,**kwargs)
       
    def interpXR(self,x,y,**kwargs) :
        ''' interpolation using xr functions '''
        useVelocity=self.__setKey('useVelocity', True, **kwargs)
        useErrors=self.__setKey('useErrors', False, **kwargs)
        x1,y1,igood=self.toInterp(x,y)
        x1xr=xr.DataArray(x1)
        y1xr=xr.DataArray(y1)
        #
        myVars=self.myVariables(useVelocity, useErrors)
        myResults=[np.full(x1.transpose().shape,np.NaN) for x in myVars]
        for myVar,i in zip(myVars,range(0,len(myVars))) :
            tmp=getattr(self,f'{myVar}').interp(x=x1xr,y=y1xr,method='linear')
            myResults[i][igood]=tmp.values.flatten()
            myResults[i]=np.reshape(myResults[i],x.shape)
        #
        return myResults
        
    def toInterp(self,x,y) :
        # flatten, do bounds check, get locations of good (inbound) points 
        x1,y1=x.flatten(),y.flatten()
        xgood=np.logical_and(x1 >= self.xx[0],x1 <= self.xx[-1])
        ygood=np.logical_and(y1 >= self.yy[0],y1 <= self.yy[-1])
        igood=np.logical_and(xgood,ygood)
        return x1[igood],y1[igood],igood
        
    def interpNP(self,x,y,**kwargs) :
        """ interpolate velocity or x at points x and y, which are 
        in m (note x,y is c-r even though data r-c)"""
        #
        useVelocity=self.__setKey('useVelocity', True, **kwargs)
        useErrors=self.__setKey('useErrors', False, **kwargs)
        #
        x1,y1,igood=self.toInterp(x,y)
        # save good points
        xy=np.array([y1,x1]).transpose() # noqa 
        #
        myVars=self.myVariables(useVelocity, useErrors)
        myResults=[np.full(x1.transpose().shape,np.NaN) for x in myVars]
        for myVar,i in zip(myVars,range(0,len(myVars))) :
            # print(self.vxInterp([0,-150000]))
            # print(myVar)
            # print(xy.shape)
            # print(getattr(self,f'{myVar}Interp'))
            # print(getattr(self,f'{myVar}Interp')(xy))
            myResults[i][igood]=getattr(self,f'{myVar}Interp')(xy)
            myResults[i]=np.reshape(myResults[i],x.shape)
        #
        return myResults
    
    def imageSize(self) :
        typeDict={'scalar' : self.x, 'velocity' : self.vx,'error': self.ex}
        ny,nx=typeDict[self.geoType].shape
        return nx,ny
    
    def computePixEdgeCornersXYM(self) :
        nx,ny=self.imageSize()
        x0,y0=self.originInM()
        dx,dy=self.pixSizeInM()
        xll,yll=x0 - dx/2,y0 - dx/2
        xur,yur= xll + nx * dx, yll + ny * dy
        xul,yul=xll,yur
        xlr,ylr=xur,yll
        corners={'ll' : {'x' :xll, 'y' : yll},'lr' : {'x' :xlr, 'y' : ylr}, 'ur' : {'x' :xur, 'y' : yur},'ul' : {'x' :xul, 'y' : yul}}
        return corners
    
    def computePixEdgeCornersLL(self) :
        corners=self.computePixEdgeCornersXYM()
        llcorners={}
        for myKey in corners.keys():
            lat,lon=self.geo.xymtoll(np.array([corners[myKey]['x']]),np.array([corners[myKey]['y']]))
            llcorners[myKey]={'lat' : lat[0],'lon' : lon[0]}
        return llcorners
        
    def getWKT_PROJ(self,epsg_code) : 
       sr=osr.SpatialReference()
       sr.ImportFromEPSG(epsg_code)
       wkt=sr.ExportToWkt()
       return wkt      
    
    def writeCloudOptGeo(self,tiffFile,suffix,epsg,gdalType,overviews=None,predictor=1,noDataDefault=None) :
        ''' write a cloudoptimized geotiff with overviews'''
        # no data info
        noDataDict={'.vx' : -2.0e9 , '.vy' : -2.0e9,'.v' : -1.0, '.ex' : -1.0, '.ey' : -1.0,'' : noDataDefault}
        #
        # use a temp mem driver for CO geo
        driver=gdal.GetDriverByName( "MEM")
        nx,ny=self.imageSize()
        dx,dy=self.geo.pixSizeInM()
        dst_ds=driver.Create('',nx,ny,1,gdalType )
        # set geometry
        tiffCorners=self.computePixEdgeCornersXYM()
        dst_ds.SetGeoTransform((tiffCorners['ul']['x'],dx,0,tiffCorners['ul']['y'],0,-dy))
        #set projection
        wkt=self.getWKT_PROJ(epsg)
        dst_ds.SetProjection(wkt)
        # set nodata
        noData=noDataDict[suffix]
        if noData != None :
            getattr(self,suffix)[np.isnan(getattr(self,suffix))]=noData
            dst_ds.GetRasterBand(1).SetNoDataValue(noData)
        # write data 
        dst_ds.GetRasterBand(1).WriteArray(np.flipud(getattr(self,suffix)))
        #
        if overviews != None :
            dst_ds.BuildOverviews('AVERAGE',overviews)
        # now copy to a geotiff - mem -> geotiff forces correct order for c opt geotiff
        dst_ds.FlushCache()
        driver=gdal.GetDriverByName( "GTiff")
        dst_ds2=driver.CreateCopy(tiffFile+suffix+'.tif',dst_ds,options=['COPY_SRC_OVERVIEWS=YES','COMPRESS=LZW',f'PREDICTOR={predictor}','TILED=YES'] )
        dst_ds2.FlushCache()
        # free memory
        dst_ds,dst_ds2=None,None
        
    def getDomain(self,epsg) :
        if epsg==None or epsg == 3413 :
            domain='greenland'
        elif epsg == 3031 :
            domain='antarctica'
        else :
            myerror('Unexpected epsg code : '+str(epsg))
        return domain
    
    
    def dataFileNames(self,fileNameBase,useVelocity=True,useErrors=False,noSpeed=True) :
        ''' compute the file names that need to be read'''
        suffixes=self.myVariables(useVelocity,useErrors,noSpeed=noSpeed)
        fileNames=[]
        if '*' not in fileNameBase :
            fileNameBase=f'{fileNameBase}.*'
        for suffix in suffixes :
            fileNames.append(f'{fileNameBase.replace("*",suffix)}.tif')
        return fileNames
        
    def readMyTiff(self,tiffFile,noData=-2.e9) :
       """ read a tiff file and return the array """
       if 1 :
           gdal.AllRegister()
           ds = gdal.Open(tiffFile)
           band=ds.GetRasterBand(1)
           arr=band.ReadAsArray()
           arr=np.flipud(arr)
           ds=None
       else :
           myerror(f"nisarVel.readMyTiff: error reading tiff file {tiffFile}")
       arr[np.isnan(arr)]=np.nan
       arr[arr <=noData]=np.nan
       return arr
   
    def readGeodatFromTiff(self,tiffFile) :
        """ Read geoinformation from a tiff file and use it to create geodat info - assumes PS coordinates"""
        try :
            gdal.AllRegister()
            ds = gdal.Open(tiffFile)
            self.sx=ds.RasterXSize
            self.sy=ds.RasterYSize
            gt=ds.GetGeoTransform()
            self.dx=abs(gt[1])
            self.dy=abs(gt[5])
            self.x0=(gt[0]+self.dx/2)
            if gt[5] < 0 :
                self.y0=(gt[3] - self.sy * self.dy + self.dy/2)
            else :
                self.y0=(gt[3] +  self.dy/2)
        except :
            myerror("Error trying to readgeodat info from tiff file: "+tiffFile)

    
    def readProduct(self,fileNameBase,useVelocity=True,useErrors=False,noSpeed=True,useXR=False) :
        ''' read in a tiff product fileNameBase.*.vx.tif'''
        #  get the values that match the type
        print(noSpeed)
        self.useXR=useXR
        minValues={'vx' : -2.e9, 'vy' : -2.e9, 'vv' : -1, 'ex' : -1, 'ey' : -1}
        fileNames=self.dataFileNames(fileNameBase,useVelocity=useVelocity,useErrors=useErrors,noSpeed=noSpeed)
        print(fileNames)
        myTypes=self.myVariables(useVelocity,useErrors,noSpeed=noSpeed)
        # loop over arrays and file names to read data
        for fileName,myType in zip(fileNames,myTypes) :
            start=datetime.now()
            if useXR :
                myArray=xr.open_rasterio(fileName)
                myArray=myArray.where(myArray > minValues[myType])
            else :
                myArray=self.readMyTiff(fileName,noData=minValues[myType])
            setattr(self,myType,myArray)
            print(f'read time {datetime.now()-start}')
        #
        self.readGeodatFromTiff(fileNames[0])
        # compute mag for velocity and errors
        if useVelocity :
            self.vv=np.sqrt(np.square(self.vx) + np.square(self.vy))
            
    def displayVel(self,fig=None) :
        if fig == None :
            sx,sy=self.sizeInPixels()
            fig=plt.figure(constrained_layout=True,figsize=(10.*sx/sy,10.))
        #   
        absmax=7000
        mxcapped=min(np.percentile(self.vv[np.isfinite(self.vv)],99),absmax)
        b= self.boundsInKm()
        axImage=fig.add_subplot(111)
        axImage.imshow(self.vv,origin='lower',vmin=0,vmax=mxcapped,extent=(b[0],b[2],b[1],b[3]))
        return axImage
    
    def sizeInPixels(self):
        return self.sx,self.sy
    
    def _toKm(func) :
        @functools.wraps(func)
        def convertKM(*args) :
            return [x*0.001 for x in func(*args)]
        return convertKM
    
    
    def sizeInM(self):
        return self.sx*self.dx,self.sy*self.dy
    
    @_toKm
    def sizeInKm(self):
        return self.sizeInM()
    
    def originInM(self):
        return self.x0,self.y0   
    @_toKm
    def originInKm(self):
        return self.originInM()
    
    def boundsInM(self):
        return self.x0,self.y0,(self.x0+(self.sx-1)*self.dx),(self.y0+(self.sy-1)*self.dy)
    @_toKm
    def boundsInKm(self):
        return self.boundsInM()

    def pixSizeInM(self):
        return self.dx,self.dy
    @_toKm
    def pixSizeInKm(self):
        return self.pixSizeInM()