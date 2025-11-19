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


# BATCH/MATRIX API

def osrm_batch_routing(coordinates_list):
    """
    Batch routing using OSRM Matrix API.

    Takes multiple origin-destination pairs and calculates travel DURATIONS
    in a SINGLE batch request instead of multiple individual requests.

    Args:
        coordinates_list: List of [origin_lat, origin_lon, dest_lat, dest_lon]

    Returns:
        List of durations in MINUTES for each pair (matches osrm_routing_machine_safe)

    Example:
        coords = [[37.56, 126.97, 37.49, 127.00], [37.57, 126.98, 37.50, 127.01]]
        durations = osrm_batch_routing(coords)
        # Returns: [15.5, 23.2] in minutes
    """
    session = requests.Session()
    retry = Retry(connect=10, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        # Format: lon1,lat1;lon2,lat2;lon3,lat3;...
        # For N coordinate pairs, we have 2N points (origin + destination for each pair)
        coords_formatted = []
        for coords in coordinates_list:
            # Origin: lon, lat (indices 1, 0)
            # Destination: lon, lat (indices 3, 2)
            coords_formatted.append(f"{coords[1]},{coords[0]}")  # origin as lon,lat
            coords_formatted.append(f"{coords[3]},{coords[2]}")  # destination as lon,lat

        coords_str = ";".join(coords_formatted)

        # OSRM matrix API endpoint
        url = f"http://127.0.0.1:5000/table/v1/driving/{coords_str}"

        # Make single batch request with 30 second timeout for large batches
        # Request DURATIONS (in seconds) not distances to match original behavior
        r = session.get(url, params={'annotations': 'duration'}, timeout=30)
        r.raise_for_status()

        response = r.json()
        durations = response['durations']

        # Extract the relevant durations (origin -> destination for each pair)
        # durations[i][j] = duration in seconds from point i to point j
        # We want durations[2k][2k+1] for each pair k, converted to minutes
        result_durations = []
        for k in range(len(coordinates_list)):
            origin_idx = k * 2
            dest_idx = (k * 2) + 1
            duration_seconds = durations[origin_idx][dest_idx]
            duration_minutes = duration_seconds / 60  # Convert to minutes like original
            result_durations.append(duration_minutes)

        return result_durations

    except Exception as e:
        # Fallback: Return estimated durations if batch request fails
        import warnings
        warnings.warn(f"Batch OSRM request failed: {e}. Using haversine fallback.")
        # Calculate distance in km, then estimate duration assuming 10 km/h average speed
        speed_km_per_hour = 10
        speed_km_per_minute = speed_km_per_hour / 60
        return [calculate_straight_distance(c[0], c[1], c[2], c[3]) / speed_km_per_minute
                for c in coordinates_list]
