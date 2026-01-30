import pandas as pd
import numpy as np

from module.osrm_routing import osrm_routing_machine_safe, osrm_batch_table
from module.utils import save_json_data



def extract_distance(distance_result):
    """
    Extract numeric distance (meters) from the osrm_routing_machine_safe result.
    Adjust this depending on your function's actual return structure.
    """
    if isinstance(distance_result, dict):
        # Example expected structure: {'routes': [{'distance': 1234.5, ...}], ...}
        try:
            return distance_result['routes'][0]['distance']
        except (KeyError, IndexError):
            # fallback or default
            return float('inf')
    elif isinstance(distance_result, (float, int)):
        return distance_result
    else:
        return float('inf')


def dispatch_main(requested_passenger, active_vehicle, empty_vehicle, simul_configs, time):
    new_trips = pd.DataFrame(columns=[
        'trip_id', 'passenger_id', 'vehicle_id', 'start_time', 'end_time', 'origin', 'destination'
    ])

    if requested_passenger.empty or empty_vehicle.empty:
        return requested_passenger, active_vehicle, empty_vehicle, new_trips

    requested_passenger = requested_passenger.sort_values('dispatch_time').reset_index(drop=True)

    # pre-collect all coordinates for batch processing optimization
    # This replaces the individual osrm_routing_machine_safe calls in the nested loop

    # 1. Collect all (vehicle -> passenger pickup) coordinates for the distance matrix
    all_pickup_coords = []
    pickup_coord_map = []  # Track (p_idx, v_idx) for each coordinate
    for p_idx, passenger in requested_passenger.iterrows():
        for v_idx, vehicle in empty_vehicle.iterrows():
            all_pickup_coords.append([
                vehicle['lat'], vehicle['lon'],
                passenger['ride_lat'], passenger['ride_lon']
            ])
            pickup_coord_map.append((p_idx, v_idx))

    # 2. Collect all (passenger ride -> alight) coordinates for dropoff distances
    all_dropoff_coords = []
    dropoff_coord_map = []  # Track p_idx for each coordinate
    for p_idx, passenger in requested_passenger.iterrows():
        all_dropoff_coords.append([
            passenger['ride_lat'], passenger['ride_lon'],
            passenger['alight_lat'], passenger['alight_lon']
        ])
        dropoff_coord_map.append(p_idx)

    # Execute batch calls
    if all_pickup_coords:
        pickup_batch_results = osrm_batch_table(all_pickup_coords)
    else:
        pickup_batch_results = []

    if all_dropoff_coords:
        dropoff_batch_results = osrm_batch_table(all_dropoff_coords)
    else:
        dropoff_batch_results = []

    # Build lookup dictionaries from batch results
    # pickup_results[(p_idx, v_idx)] -> raw result (to be passed to extract_distance)
    pickup_results = {}
    for i, (p_idx, v_idx) in enumerate(pickup_coord_map):
        pickup_results[(p_idx, v_idx)] = pickup_batch_results[i] if i < len(pickup_batch_results) else None

    # dropoff_results[p_idx] -> raw result (to be passed to extract_distance)
    dropoff_results = {}
    for i, p_idx in enumerate(dropoff_coord_map):
        dropoff_results[p_idx] = dropoff_batch_results[i] if i < len(dropoff_batch_results) else None

    # only data source changed from individual calls to batch results

    assigned_idx = []
    for p_idx, passenger in requested_passenger.iterrows():
        min_distance = float('inf')
        chosen_vehicle_idx = None

        for v_idx, vehicle in empty_vehicle.iterrows():
            # use batch pre-computed result
            dist_result = pickup_results.get((p_idx, v_idx))
            distance = extract_distance(dist_result)

            if distance < min_distance:
                min_distance = distance
                chosen_vehicle_idx = v_idx

        if chosen_vehicle_idx is not None:
            vehicle = empty_vehicle.loc[chosen_vehicle_idx].copy()

            # pre-computed result winning comparison
            distance_to_pickup = extract_distance(pickup_results.get((p_idx, chosen_vehicle_idx)))
            # se pre-computed result instead of extracting from osrm_routing_machine_safe()
            distance_to_dropoff = extract_distance(dropoff_results.get(p_idx))

            pickup_time = time + distance_to_pickup / 60  # meters to minutes assumption
            dropoff_time = pickup_time + distance_to_dropoff / 60 + simul_configs.get('add_board_time', 0)

            vehicle['lat'] = passenger['alight_lat']
            vehicle['lon'] = passenger['alight_lon']
            vehicle['work_start'] = time
            vehicle['work_end'] = dropoff_time

            active_vehicle = pd.concat([active_vehicle, vehicle.to_frame().T])
            empty_vehicle = empty_vehicle.drop(index=chosen_vehicle_idx)

            assigned_idx.append(p_idx)

            new_trips = pd.concat([new_trips, pd.DataFrame([{
                'trip_id': f"{time}_{passenger['ID']}_{vehicle['vehicle_id']}",
                'passenger_id': passenger['ID'],
                'vehicle_id': vehicle['vehicle_id'],
                'start_time': pickup_time,
                'end_time': dropoff_time,
                'origin': (passenger['ride_lat'], passenger['ride_lon']),
                'destination': (passenger['alight_lat'], passenger['alight_lon'])
            }])])

    requested_passenger = requested_passenger.drop(index=assigned_idx).reset_index(drop=True)
    active_vehicle = active_vehicle.reset_index(drop=True)
    empty_vehicle = empty_vehicle.reset_index(drop=True)

    return requested_passenger, active_vehicle, empty_vehicle, new_trips