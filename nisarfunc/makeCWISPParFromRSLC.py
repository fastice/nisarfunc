import copy
import nisarhdf
import numpy as np
import json

template = {
    'title': {'value': '', 'units': ''},
    'sensor': {'value': 'NISAR', 'units': ''},
    'date': {'value': '26 AUG 2025', 'units': ''},
    'start_time': {'value': 30836.5981, 'units': 's'}, #xx
    "center_time": {"value": 30905.71932, "units": "s"}, #xx
    "end_time": {"value": 30974.84254, "units": "s"},
    "line_header_size": {"value": 0, "units": ""},
    "range_samples": {"value": 67816, "units": ""}, #--
    "azimuth_lines": {"value": 67254, "units": ""}, #--
    "range_looks": {"value": 1, "units": ""},
    "azimuth_looks": {"value": 1, "units": ""},
    "range_scale_factor": {"value": 1.0000000, "units": ""},
    "azimuth_scale_factor": {"value": 1.0000000, "units": ""},
    "center_latitude": {"value": 81.4618233, "units": "degrees"},#xx
    "center_longitude": {"value": -7.2084689, "units": "degrees"},#xx
    "heading": {"value": 0.0000000, "units": "degrees"},
    "range_pixel_spacing": {"value": 2.329562, "units": "m"},
    "azimuth_pixel_spacing": {"value": 13.879420, "units": "m"},
    "near_range_slc": {"value": 803891.8416, "units": "m"}, #xx
    "center_range_slc": {"value": 882881.4651, "units": "m"}, #xxx
    "far_range_slc": {"value": 961871.0886, "units": "m"},#xxx
    "incidence_angle": {"value": 38.9757, "units": "degrees"},
    "azimuth_deskew": {"value": None, "units": ""},
    "azimuth_angle": {"value": -90.0000, "units": "degrees"},
    "radar_frequency": {"value": 5.4050005e+09, "units": "Hz"}, #xx
    "adc_sampling_rate": {"value": 6.4345241e+07, "units": "Hz"}, #xx
    "chirp_bandwidth": {"value": 5.65000e+07, "units": "Hz"}, #xx
    "prf": {"value": 486.48631, "units": "Hz"},
    "azimuth_proc_bandwidth": {"value": 327.00004, "units": "Hz"}, #xx
    "doppler_polynomial": { #xx
        "value": [-24.84454, -9.90517e-06, 9.84114e-10, 0.00000e+00],
        "units": ["Hz Hz/m Hz/m^2 Hz/m^3"],
    },
    "receiver_gain": {"value": 0.0000, "units": "dB"},
    "calibration_gain": {"value": 0.0000, "units": "dB"},
    "sar_to_earth_center": {"value": 0, "units": "m"}, #xx
    "earth_radius_below_sensor": {"value": 0, "units": "m"}, #xx
    "earth_semi_major_axis": {"value": 6378137.0000, "units": "m"},
    "earth_semi_minor_axis": {"value": 6356752.3141, "units": "m"},
    "number_of_state_vectors": {"value": 42, "units": ""},
    "time_of_first_state_vector": {"value": 30762.00000, "units": "s"},
    "state_vector_interval": {"value": 10.00000, "units": "s"},
}

def processOneToOne(geoProperties, myCW):
    oneToOne = {'title': 'ImageName',
                'range_looks': 'NumberRangeLooks',
                'azimuth_looks': 'NumberAzimuthLooks',
                'number_of_state_vectors': 'NumberOfStateVectors',
                'time_of_first_state_vector': 'TimeOfFirstStateVector',
                'state_vector_interval': 'StateVectorInterval',
                'range_pixel_spacing': 'SLCRangePixelSize',
                'azimuth_pixel_spacing': 'SLCAzimuthPixelSize',
                'prf': 'PRF',
                'incidence_angle': 'MLIncidenceCenter'}
    for key in oneToOne:
        myCW[key] = geoProperties[oneToOne[key]]
    

            
def readGeodat(geodat):
    with open(geodat, "r") as f:
        gj = json.load(f)

    # features list
    #print(gj.keys())
    features = gj["properties"]
    #for f in gj["properties"]:
     #   print(f, gj["properties"][f])
    return gj['properties']

def processStateVectors(myRSLC, myCW):
    myCW["number_of_state_vectors"]['value'] = myRSLC.orbit.NumberOfStateVectors
    myCW["time_of_first_state_vector"]['value'] = myRSLC.orbit.TimeOfFirstStateVector
    myCW["state_vector_interval"]['value'] = myRSLC.orbit.StateVectorInterval
    for i, pos, vel in zip(range(1, myRSLC.orbit.NumberOfStateVectors + 1),
                           myRSLC.orbit.position, myRSLC.orbit.velocity):
        myCW[f'state_vector_position_{i}'] = {'value': pos, 'units': 'm m m'}
        myCW[f'state_vector_velocity_{i}'] = {'value': vel, 'units': 'm/s m/s m/s'}
        
def hms_to_seconds_of_day(timestr: str) -> float:
    h, m, s = timestr.split()
    return int(h)*3600 + int(m)*60 + float(s)
    
def processTime(myRSLC, myCW):
    centerTime = (myRSLC.SLCFirstZeroDopplerTime + myRSLC.SLCLastZeroDopplerTime) * 0.5
    myCW['start_time']['value'] = myRSLC.SLCFirstZeroDopplerTime 
    myCW['center_time']['value'] = centerTime
    myCW['end_time']['value'] =  myRSLC.SLCLastZeroDopplerTime
    
def processSLCSize(myRSLC, myCW):
    myCW["range_samples"] = {"value": myRSLC.SLCRangeSize,
                             "units": ""}
    myCW["azimuth_lines"] = {"value": myRSLC.SLCAzimuthSize, 
                             "units": ""}

def processTitle(myCW, RSLCFile):
    myCW['title']['value'] = RSLCFile

def processCenterLL(myRSLC, myCW):
    myCW['center_latitude']['value'] = myRSLC.centerLat
    myCW['center_longitude']['value'] = myRSLC.centerLon

def processPixelSpacing(myRSLC, myCW):
    myCW['range_pixel_spacing']['value'] = myRSLC.SLCRangePixelSize
    myCW['azimuth_pixel_spacing']['value'] = myRSLC.SLCAzimuthPixelSize

def processRange(myRSLC, myCW):
    myCW['near_range_slc']['value'] = myRSLC.SLCNearRange
    myCW['center_range_slc']['value'] = (myRSLC.SLCNearRange + myRSLC.SLCFarRange) * 0.5
    myCW['far_range_slc']['value'] = myRSLC.SLCFarRange

def processCenterFreq(myRSLC, myCW):
    c = 299_792_458  # m/s
    myCW['radar_frequency']['value'] = c / myRSLC.Wavelength
    myCW['adc_sampling_rate']['value'] = c / (2.0 * myRSLC.SLCRangePixelSize)
    myCW['chirp_bandwidth']['value'] = myRSLC.rangeBandwidth

def processPRF(myRSLC, myCW):
    myCW['prf']['value'] = myRSLC.PRF
    myCW['azimuth_proc_bandwidth']['value'] = \
        np.array(myRSLC.h5[myRSLC.product]['swaths']['frequencyA']['processedAzimuthBandwidth']).item()

def processDoppler(myRSLC, myCW):
    params = myRSLC.h5[myRSLC.product]['metadata']['processingInformation']['parameters']
    # uses mean doppler for now
    myCW['doppler_polynomial']['value'] = \
        [np.mean(params['frequencyA']['dopplerCentroid']), 0., 0., 0.]

def printCWInSARFile(myCW, cwFileName):
    with open(cwFileName, 'w') as fp:
        for key in myCW:
            #print(key, myCW[key])
            x = myCW[key]['value']
            if isinstance(x, np.ndarray) or isinstance(x, list):
                print(key, ':', ' '.join([f'{a}' for a in x]),  myCW[key]['units'], file=fp)
            else:
                print(key, ':', x,  myCW[key]['units'], file=fp)

            
            #print(type(myCW[key]['value']))
def makeCWISPParFromRSLC(RSLCFile, cwFileName):
    myRSLC = nisarhdf.nisarRSLCHDF()
    myRSLC.openHDF(RSLCFile, noLoadData=True)
    myCW = copy.deepcopy(template)
    processTitle(myCW, RSLCFile)
    myCW['date']['value'] = myRSLC.datetime.strftime("%d %b %Y").upper()
    processTime(myRSLC, myCW)
    #processOneToOne(geoProperties, myCW)
    processSLCSize(myRSLC, myCW)
    processCenterLL(myRSLC, myCW)
    processPixelSpacing(myRSLC, myCW)
    processRange(myRSLC, myCW)
    processCenterFreq(myRSLC, myCW)
    myCW["incidence_angle"]['value'] = np.degrees(myRSLC.SLCIncidenceCenter)
    processPRF(myRSLC, myCW)
    processDoppler(myRSLC, myCW)
    processStateVectors(myRSLC, myCW)
    #
    print('done processing, now writing CW file ... ')
    printCWInSARFile(myCW, cwFileName)
    return myCW
