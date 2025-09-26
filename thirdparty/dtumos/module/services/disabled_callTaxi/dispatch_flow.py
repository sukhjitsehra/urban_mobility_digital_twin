import pandas as pd
import numpy as np

from module.osrm_routing import osrm_routing_machine_safe
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

    assigned_idx = []
    for p_idx, passenger in requested_passenger.iterrows():
        min_distance = float('inf')
        chosen_vehicle_idx = None

        for v_idx, vehicle in empty_vehicle.iterrows():
            dist_result = osrm_routing_machine_safe([
                vehicle['lat'], vehicle['lon'],
                passenger['ride_lat'], passenger['ride_lon']
            ])
            distance = extract_distance(dist_result)

            if distance < min_distance:
                min_distance = distance
                chosen_vehicle_idx = v_idx

        if chosen_vehicle_idx is not None:
            vehicle = empty_vehicle.loc[chosen_vehicle_idx].copy()

            distance_to_pickup = extract_distance(osrm_routing_machine_safe([
                vehicle['lat'], vehicle['lon'],
                passenger['ride_lat'], passenger['ride_lon']
            ]))
            distance_to_dropoff = extract_distance(osrm_routing_machine_safe([
                passenger['ride_lat'], passenger['ride_lon'],
                passenger['alight_lat'], passenger['alight_lon']
            ]))

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