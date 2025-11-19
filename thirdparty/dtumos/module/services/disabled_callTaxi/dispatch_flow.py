import pandas as pd
import numpy as np

from module.osrm_routing import osrm_routing_machine_safe, osrm_batch_routing
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


def build_distance_matrix_batch(passengers_df, vehicles_df):
    """
    Build distance matrix using batch OSRM API.

    Args:
        passengers_df: DataFrame with 'ride_lat', 'ride_lon' columns
        vehicles_df: DataFrame with 'lat', 'lon' columns

    Returns:
        2D numpy array where [i, j] = distance from vehicle i to passenger j (in meters)
    """
    if passengers_df.empty or vehicles_df.empty:
        return np.array([])

    # Build list of all vehicle->passenger coordinate pairs
    coords_list = []
    for _, vehicle in vehicles_df.iterrows():
        for _, passenger in passengers_df.iterrows():
            # Format: [origin_lat, origin_lon, dest_lat, dest_lon]
            coords_list.append([
                vehicle['lat'], vehicle['lon'],
                passenger['ride_lat'], passenger['ride_lon']
            ])

    # Make single batch call for all distances
    distances = osrm_batch_routing(coords_list)

    # Reshape into matrix: rows = vehicles, cols = passengers
    n_vehicles = len(vehicles_df)
    n_passengers = len(passengers_df)
    distance_matrix = np.array(distances).reshape(n_vehicles, n_passengers)

    return distance_matrix


def dispatch_main(requested_passenger, active_vehicle, empty_vehicle, simul_configs, time):
    new_trips = pd.DataFrame(columns=[
        'trip_id', 'passenger_id', 'vehicle_id', 'start_time', 'end_time', 'origin', 'destination'
    ])

    if requested_passenger.empty or empty_vehicle.empty:
        return requested_passenger, active_vehicle, empty_vehicle, new_trips

    requested_passenger = requested_passenger.sort_values('dispatch_time').reset_index(drop=True)

    # OPTIMIZATION: Build distance matrix with single batch API call
    # Instead of nested loop with individual OSRM calls
    distance_matrix = build_distance_matrix_batch(requested_passenger, empty_vehicle)

    assigned_idx = []
    matched_pairs = []  # Store (passenger_idx, vehicle_idx, distance) for batch processing
    available_vehicles = set(range(len(empty_vehicle)))  # Track which vehicles are still available

    # CRITICAL: Process passengers in order, assigning each to best AVAILABLE vehicle
    # This maintains the original greedy assignment logic
    for p_idx in range(len(requested_passenger)):
        if not available_vehicles:
            break  # No more vehicles available

        # Get distances from all vehicles to this passenger
        passenger_distances = distance_matrix[:, p_idx]

        # Find best AVAILABLE vehicle (not already assigned)
        best_distance = float('inf')
        best_vehicle_idx = None

        for v_idx in available_vehicles:
            if passenger_distances[v_idx] < best_distance:
                best_distance = passenger_distances[v_idx]
                best_vehicle_idx = v_idx

        if best_vehicle_idx is not None and best_distance < float('inf'):
            matched_pairs.append((p_idx, best_vehicle_idx, best_distance))
            available_vehicles.remove(best_vehicle_idx)  # Remove from available pool
            assigned_idx.append(p_idx)

    # OPTIMIZATION: Batch calculate pickup and dropoff distances for all matched pairs
    if matched_pairs:
        # Build coordinate lists for pickup and dropoff distances
        pickup_coords = []
        dropoff_coords = []

        for p_idx, v_idx, _ in matched_pairs:
            passenger = requested_passenger.iloc[p_idx]
            vehicle = empty_vehicle.iloc[v_idx]

            # Pickup distance: vehicle location -> passenger pickup location
            pickup_coords.append([
                vehicle['lat'], vehicle['lon'],
                passenger['ride_lat'], passenger['ride_lon']
            ])

            # Dropoff distance: passenger pickup -> passenger dropoff
            dropoff_coords.append([
                passenger['ride_lat'], passenger['ride_lon'],
                passenger['alight_lat'], passenger['alight_lon']
            ])

        # Single batch call for all pickup and dropoff distances
        all_coords = pickup_coords + dropoff_coords
        all_distances = osrm_batch_routing(all_coords)

        # Split results
        n_pairs = len(matched_pairs)
        pickup_distances = all_distances[:n_pairs]
        dropoff_distances = all_distances[n_pairs:]

        # Process each matched pair with pre-calculated distances
        assigned_vehicle_indices = []
        for idx, (p_idx, v_idx, _) in enumerate(matched_pairs):
            passenger = requested_passenger.iloc[p_idx]
            vehicle = empty_vehicle.iloc[v_idx].copy()

            duration_to_pickup = pickup_distances[idx]  # Actually duration in minutes from batch API
            duration_to_dropoff = dropoff_distances[idx]  # Actually duration in minutes from batch API

            pickup_time = time + duration_to_pickup
            dropoff_time = pickup_time + duration_to_dropoff + simul_configs.get('add_board_time', 0)

            # Set passenger info on vehicle (required by update_vehicle to track state)
            vehicle['P_ID'] = passenger['ID']
            vehicle['P_ride_lat'] = passenger['ride_lat']
            vehicle['P_ride_lon'] = passenger['ride_lon']
            vehicle['P_alight_lat'] = passenger['alight_lat']
            vehicle['P_alight_lon'] = passenger['alight_lon']
            vehicle['P_request_time'] = passenger['ride_time']
            vehicle['P_disembark_time'] = dropoff_time

            active_vehicle = pd.concat([active_vehicle, vehicle.to_frame().T])
            assigned_vehicle_indices.append(v_idx)

            new_trips = pd.concat([new_trips, pd.DataFrame([{
                'trip_id': f"{time}_{passenger['ID']}_{vehicle['vehicle_id']}",
                'passenger_id': passenger['ID'],
                'vehicle_id': vehicle['vehicle_id'],
                'start_time': pickup_time,
                'end_time': dropoff_time,
                'origin': (passenger['ride_lat'], passenger['ride_lon']),
                'destination': (passenger['alight_lat'], passenger['alight_lon'])
            }])])

        # Remove assigned vehicles
        empty_vehicle = empty_vehicle.drop(empty_vehicle.index[assigned_vehicle_indices]).reset_index(drop=True)

    requested_passenger = requested_passenger.drop(index=assigned_idx).reset_index(drop=True)
    active_vehicle = active_vehicle.reset_index(drop=True)

    return requested_passenger, active_vehicle, empty_vehicle, new_trips