import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from module.simulator_helper import (
    generate_path_to_save,
    extract_selector,
    dispatch_selector,
    crop_data_by_timerange,
    base_data,
    checking_progress,
    save_json_data
)

from module.simulator_object_update import update_passenger, update_vehicle

from module.simulator_object_update import update_passenger  # ✅ correct


class Simulator:

    def __init__(self, raw_data=None, passengers=None, vehicles=None, configs=None, save_path=None):
        if configs is None:
            raise ValueError("Please provide the configs data.")
        self.configs = configs

        # Accept explicit save_path or generate base path
        if save_path is not None:
            base_save_path = Path(save_path)
        else:
            base_save_path = Path(generate_path_to_save(self.configs['path'], self.configs['additional_path']))
        base_save_path.mkdir(parents=True, exist_ok=True)

        # Create numbered simulation folder
        sim_number = len([f for f in base_save_path.iterdir() if f.is_dir()]) + 1
        self.save_folder = base_save_path / f"simulation_{sim_number}"
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.configs['save_path'] = str(self.save_folder)
        print(f"[Simulator] Save path: {self.configs['save_path']}")

        # Problem selector
        self.extract_main = extract_selector(self.configs["problem"])
        self.dispatch_main = dispatch_selector(self.configs["problem"])

        # Object data
        self.raw_data = raw_data
        self.passengers = passengers
        self.vehicles = vehicles

        if self.raw_data is not None:
            self.passengers, self.vehicles, YMD = self.extract_main(self.raw_data, self.configs)
            self.configs['YMD'] = YMD
        else:
            self.configs['YMD'] = pd.Timestamp('2019-04-09 00:00:00')

        # Preprocess data
        self.passengers, self.vehicles = crop_data_by_timerange(self.passengers, self.vehicles, self.configs)

        # Generate simulation base data
        self.active_vehicle, self.empty_vehicle, self.requested_passenger, self.fail_passenger, self.simulation_record = base_data()

        # Initialize trips dataframe
        self.trips = pd.DataFrame(columns=['trip_id', 'passenger_id', 'vehicle_id', 'start_time', 'end_time', 'origin', 'destination'])

    def run(self):
        start_time, end_time = self.configs['time_range']
        print(f"[Simulator] Starting simulation from {start_time} to {end_time}")

        for time in tqdm(range(start_time, end_time)):
            # Update passengers and vehicles
            self.requested_passenger, self.fail_passenger, self.passengers = update_passenger(
                self.requested_passenger,
                self.fail_passenger,
                self.passengers,
                self.configs,
                time
            )
            self.active_vehicle, self.empty_vehicle, self.vehicles = update_vehicle(
                self.active_vehicle,
                self.empty_vehicle,
                self.vehicles,
                self.configs,
                time
            )

            # Dispatch
            if len(self.requested_passenger) > 0 and len(self.empty_vehicle) > 0:
                # Ensure dispatch always returns 4 values
                result = self.dispatch_main(
                    self.requested_passenger,
                    self.active_vehicle,
                    self.empty_vehicle,
                    self.configs,
                    time
                )
                # Handle 3-value return from old code
                if len(result) == 3:
                    self.requested_passenger, self.active_vehicle, self.empty_vehicle = result
                    new_trips = pd.DataFrame(columns=self.trips.columns)
                else:
                    self.requested_passenger, self.active_vehicle, self.empty_vehicle, new_trips = result

                if new_trips is not None and not new_trips.empty:
                    self.trips = pd.concat([self.trips, new_trips], ignore_index=True)

            # Record simulation
            self.simulation_record = checking_progress(
                self.simulation_record,
                time,
                self.requested_passenger,
                self.fail_passenger,
                self.empty_vehicle,
                self.active_vehicle,
                self.configs
            )

        # Save results
        save_path = self.save_folder
        print(f"[Simulator] Saving simulation results to {save_path}")
        self.simulation_record.to_csv(save_path / 'record.csv', index=False)

        # Convert DataFrames to list of dicts for JSON serialization
        passengers_list = self.passengers.to_dict('records') if isinstance(self.passengers, pd.DataFrame) else self.passengers
        trips_list = self.trips.to_dict('records') if isinstance(self.trips, pd.DataFrame) else self.trips

        save_json_data(passengers_list, save_path, 'passenger_marker')
        save_json_data(trips_list, save_path, 'trip')
        print("[Simulator] Simulation completed and results saved successfully!")
