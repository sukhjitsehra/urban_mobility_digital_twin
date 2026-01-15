# osrm_routing.py
from .utils import calculate_straight_distance
import numpy as np
import itertools
import requests
import polyline
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
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


#######################
# BATCH OPTIMIZATION  #
#######################
def osrm_batch_table(coordinates_list):
    """
    Batch routing using OSRM Table API.
    Returns dict with 'distances' (meters) and 'durations' (seconds) for each OD pair.

    Args:
        coordinates_list: List of [origin_lat, origin_lon, dest_lat, dest_lon]

    Returns:
        List of dicts matching osrm_routing_machine_safe output format:
        [{'distance': meters, 'duration': minutes}, ...]
    """
    # Chunk into batches of CHUNK_SIZE OD pairs (1000 coordinates)
    CHUNK_SIZE = 50
    MAX_WORKERS = 10  # Match OSRM --threads=10
    if len(coordinates_list) > CHUNK_SIZE:
        # Split into chunks
        chunks = []
        for i in range(0, len(coordinates_list), CHUNK_SIZE):
            chunks.append(coordinates_list[i:i + CHUNK_SIZE])

        num_workers = min(len(chunks), MAX_WORKERS)
        print(f"[OSRM Batch] Chunking {len(coordinates_list)} OD pairs into {len(chunks)} batches, processing {num_workers} in parallel")

        # Process chunks in parallel
        results_by_index = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {executor.submit(osrm_batch_table, chunk): i for i, chunk in enumerate(chunks)}
            for future in future_to_index:
                chunk_index = future_to_index[future]
                try:
                    results_by_index[chunk_index] = future.result()
                except Exception as e:
                    print(f"[OSRM Batch] Chunk {chunk_index} failed ({e}), using fallback")
                    results_by_index[chunk_index] = [osrm_routing_machine_safe(coords) for coords in chunks[chunk_index]]

        # Reassemble results in correct order
        all_results = []
        for i in range(len(chunks)):
            all_results.extend(results_by_index[i])
        return all_results

    session = requests.Session()
    retry = Retry(connect=10, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        # Build coordinate string for OSRM table API
        # Format: lon1,lat1;lon2,lat2;...
        coords_formatted = []
        for coords in coordinates_list:
            coords_formatted.append(f"{coords[1]},{coords[0]}")  # origin
            coords_formatted.append(f"{coords[3]},{coords[2]}")  # destination

        coords_str = ";".join(coords_formatted)
        url = f"http://127.0.0.1:5000/table/v1/driving/{coords_str}"

        r = session.get(url, params={'annotations': 'distance,duration'}, timeout=30)
        r.raise_for_status()
        response = r.json()

        results = []
        for k in range(len(coordinates_list)):
            origin_idx = k * 2
            dest_idx = k * 2 + 1
            distance = response['distances'][origin_idx][dest_idx]
            duration_seconds = response['durations'][origin_idx][dest_idx]

            # Handle None values (unreachable routes) - use fallback calculation
            if distance is None or duration_seconds is None:
                coords = coordinates_list[k]
                # Fallback: calculate straight-line distance in meters
                distance = calculate_straight_distance(coords[0], coords[1], coords[2], coords[3]) * 1000
                # Estimate duration: assume 30 km/h average speed
                duration = (distance / 1000) / 30 * 60  # Convert to minutes
            else:
                duration = duration_seconds / 60  # Convert seconds to minutes

            results.append({'distance': distance, 'duration': duration})

        return results

    except Exception as e:
        # Fallback to individual calls
        print(f"[OSRM Batch] Batch API failed ({e}), falling back to {len(coordinates_list)} individual calls")
        return [osrm_routing_machine_safe(coords) for coords in coordinates_list]
