from .simulator_helper import save_json_data
import pandas as pd
import numpy as np

###########################
# Passenger status update #
###########################
def update_passenger(requested_passenger, fail_passenger, passenger, simul_configs, time):
    fail_time = simul_configs['fail_time']
    save_path = simul_configs['save_path']
    
    current_requested_passenger = passenger.loc[passenger['ride_time'] == time]
    passenger = passenger.loc[passenger['ride_time'] != time].reset_index(drop=True)
    
    if len(requested_passenger) > 0:
        requested_passenger['dispatch_time'] += 1
        
        current_fail_passenger = requested_passenger.loc[requested_passenger['dispatch_time'] >= fail_time]
        fail_passenger = pd.concat([fail_passenger, current_fail_passenger]).reset_index(drop=True)
        
        if len(current_fail_passenger) > 0:
            requested_passenger = requested_passenger.loc[requested_passenger['dispatch_time'] < fail_time].reset_index(drop=True)
            
            fail_marker = [
                {
                    'passenger_id': row['ID'], 'status': 0,
                    'location': [row['ride_lon'], row['ride_lat']],
                    'timestamp': [row['ride_time'], row['ride_time'] + row['dispatch_time']]
                }
                for _, row in current_fail_passenger.iterrows()
            ]
            save_json_data(fail_marker, save_path, file_name='passenger_marker')
            del fail_marker
    
    requested_passenger = pd.concat([requested_passenger, current_requested_passenger]).reset_index(drop=True)
    return requested_passenger, fail_passenger, passenger

#########################
# Vehicle status update #
#########################
def update_vehicle(active_vehicle, empty_vehicle, vehicle, simul_configs, time):
    save_path = simul_configs['save_path']
    
    # Start work vehicles
    current_start_vehicle = vehicle.loc[vehicle['work_start'] == time]
    if len(current_start_vehicle) > 0:
        cols = ['vehicle_id', 'work_end', 'temporary_stopTime', 'lat', 'lon']
        if 'cartype' in current_start_vehicle.columns:
            cols.insert(1, 'cartype')
        current_start_vehicle = current_start_vehicle[cols]
        current_start_vehicle['temporary_stopTime'] = time
        current_start_vehicle[['P_ID', 'P_ride_lat', 'P_ride_lon', 'P_alight_lat', 'P_alight_lon', 'P_request_time', 'P_disembark_time']] = np.nan
        empty_vehicle = pd.concat([empty_vehicle, current_start_vehicle]).reset_index(drop=True)
        vehicle = vehicle.loc[vehicle['work_start'] != time].reset_index(drop=True)
    
    # Drop off passengers
    if len(active_vehicle) > 0:
        current_empty_vehicle = active_vehicle.loc[active_vehicle['P_disembark_time'] <= time].copy()
        if len(current_empty_vehicle) > 0:
            current_empty_vehicle['lat'] = current_empty_vehicle['P_alight_lat']
            current_empty_vehicle['lon'] = current_empty_vehicle['P_alight_lon']
            current_empty_vehicle['temporary_stopTime'] = current_empty_vehicle['P_disembark_time']
            current_empty_vehicle[['P_ID', 'P_ride_lat', 'P_ride_lon', 'P_alight_lat', 'P_alight_lon', 'P_disembark_time']] = np.nan
            empty_vehicle = pd.concat([empty_vehicle, current_empty_vehicle]).reset_index(drop=True)
            active_vehicle = active_vehicle.loc[active_vehicle['P_disembark_time'] > time].reset_index(drop=True)
    
    # End work vehicles
    if len(empty_vehicle) > 0:
        end_vehicle = empty_vehicle.loc[empty_vehicle['work_end'] < time + 5]
        empty_vehicle = empty_vehicle.loc[empty_vehicle['work_end'] >= time + 5].reset_index(drop=True)
        
        if len(end_vehicle) > 0:
            end_marker = [
                {
                    'vehicle_id': row['vehicle_id'],
                    'cartype': row['cartype'] if 'cartype' in row else None,
                    'location': [row['lon'], row['lat']],
                    'timestamp': [row['temporary_stopTime'], time]
                }
                for _, row in end_vehicle.iterrows() if not pd.isna(row['temporary_stopTime'])
            ]
            save_json_data(end_marker, save_path, file_name='vehicle_marker')
            del end_marker
    
    return active_vehicle, empty_vehicle, vehicle

#########################
# Dispatch function fix #
#########################
def dispatch_in_order(requested_passenger, active_vehicle, empty_vehicle, simul_configs, time):
    # Example dispatch logic: assign first available passenger to first empty vehicle
    new_trips = pd.DataFrame(columns=['trip_id', 'passenger_id', 'vehicle_id', 'start_time', 'end_time', 'origin', 'destination'])
    
    if len(requested_passenger) > 0 and len(empty_vehicle) > 0:
        # Simple example: pair first passenger with first vehicle
        p = requested_passenger.iloc[0]
        v = empty_vehicle.iloc[0]
        trip_id = len(new_trips) + 1
        new_trips = pd.DataFrame([{
            'trip_id': trip_id,
            'passenger_id': p['ID'],
            'vehicle_id': v['vehicle_id'],
            'start_time': time,
            'end_time': time + 15,  # example duration
            'origin': [p['ride_lon'], p['ride_lat']],
            'destination': [p['ride_lon'] + 0.01, p['ride_lat'] + 0.01]  # example
        }])
        requested_passenger = requested_passenger.iloc[1:].reset_index(drop=True)
        empty_vehicle = empty_vehicle.iloc[1:].reset_index(drop=True)
        active_vehicle = pd.concat([active_vehicle, v.to_frame().T]).reset_index(drop=True)
    
    return requested_passenger, active_vehicle, empty_vehicle, new_trips
