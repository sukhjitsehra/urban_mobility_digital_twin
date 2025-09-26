# osrm_routing.py
from .utils import calculate_straight_distance
import numpy as np
import itertools
import requests
import polyline
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import warnings 
warnings.filterwarnings('ignore')

#############
# OSRM base #
#############
def get_res(point):
    session = requests.Session()
    retry = Retry(connect=10, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    overview = '?overview=full'
    loc = f"{point[1]},{point[0]};{point[3]},{point[2]}"  # lon,lat,lon,lat
    url = "http://127.0.0.1:5000/route/v1/driving/"
    try:
        r = session.get(url + loc + overview, timeout=10)
        r.raise_for_status()
        res = r.json()
        return res, 'ok'
    except:
        # fallback: straight-line distance
        distance = calculate_straight_distance(point[0], point[1], point[2], point[3]) * 1000
        route = [[point[1], point[0]], [point[3], point[2]]]
        speed_km = 10
        speed = (speed_km * 1000 / 60)
        duration = distance / speed
        timestamp = [0, duration]
        result = {'route': route, 'timestamp': timestamp, 'duration': duration, 'distance': distance}
        return result, 'defined'

##############################
# Extract duration, distance #
##############################
def extract_duration_distance(res):
    duration = res['routes'][0]['duration']/60
    distance = res['routes'][0]['distance']
    return duration, distance

#################
# Extract route #
#################
def extract_route(res):
    route = polyline.decode(res['routes'][0]['geometry'])
    route = [[pt[1], pt[0]] for pt in route]
    return route

#####################
# Extract timestamp #
#####################
def extract_timestamp(route, duration):
    rt = np.array(route)
    rt = np.hstack([rt[:-1, :], rt[1:, :]])
    per = calculate_straight_distance(rt[:,1], rt[:,0], rt[:,3], rt[:,2])
    per = per / np.sum(per)
    timestamp = np.hstack([np.array([0]), per * duration])
    timestamp = list(itertools.accumulate(timestamp))
    return timestamp

########
# MAIN #
########
def osrm_routing_machine_safe(OD_coords):
    """
    Returns a dict with route, timestamp, duration, distance.
    Never returns None, always safe for dispatch.
    """
    osrm_base, status = get_res(OD_coords)
    if status == 'defined' or osrm_base is not None:
        try:
            if status != 'defined':
                duration, distance = extract_duration_distance(osrm_base)
                route = extract_route(osrm_base)
                timestamp = extract_timestamp(route, duration)
            else:
                duration = osrm_base['duration']
                distance = osrm_base['distance']
                route = osrm_base['route']
                timestamp = osrm_base['timestamp']

            result = {'route': route, 'timestamp': timestamp, 'duration': duration, 'distance': distance}

            if np.isnan(result['timestamp'][-1]):
                result['timestamp'][-1] = 0.01
                result['duration'] = 0.01

            return result
        except Exception as e:
            # fallback
            return {'route': [[OD_coords[1], OD_coords[0]], [OD_coords[3], OD_coords[2]]],
                    'timestamp':[0, 0.01], 'duration':0.01, 'distance':calculate_straight_distance(OD_coords[0], OD_coords[1], OD_coords[2], OD_coords[3])*1000}
    else:
        return {'route': [[OD_coords[1], OD_coords[0]], [OD_coords[3], OD_coords[2]]],
                'timestamp':[0, 0.01], 'duration':0.01, 'distance':calculate_straight_distance(OD_coords[0], OD_coords[1], OD_coords[2], OD_coords[3])*1000}
