import random
import eventlet
import json
import os
from datetime import datetime
from Model.simulation_state import SimulationState


class Simulation:
    def __init__(self, system, socketio, interval=10, data_file='analysis_output/hourly_request_stats.json'):
        self.system = system
        self.socketio = socketio
        self.interval = interval
        self.running = False

        # Load hourly request data
        self.hourly_request_data = self._load_hourly_request_data(data_file)

        # Prepare origin-destination pairs with probabilities
        self.od_pairs_by_hour = self._load_od_pairs()

    def _load_hourly_request_data(self, data_file):
        """
        Load hourly request statistics from JSON file.

        Args:
            data_file (str): Path to the hourly request stats JSON file

        Returns:
            dict: Hourly request statistics
        """
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Hourly request data file {data_file} not found. Falling back to random generation.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {data_file}. Falling back to random generation.")
            return {}

    def _load_od_pairs(self, data_file='analysis_output/od_pairs_by_time.json'):
        """
        Load origin-destination pairs for each hour.

        Args:
            data_file (str): Path to the OD pairs JSON file

        Returns:
            dict: OD pairs for each hour
        """
        try:
            with open(data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: OD pairs file {data_file} not found. Falling back to random generation.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {data_file}. Falling back to random generation.")
            return {}

    def start(self):
        """Starts the simulation in the background."""
        self.running = True
        # Set state to RUNNING here
        self.system.transport_manager.set_state(SimulationState.RUNNING)
        eventlet.spawn_n(self._run_loop)

    def stop(self):
        """Stops the simulation loop."""
        self.running = False
        # Set state to READY here
        self.system.transport_manager.set_state(SimulationState.READY)

    def is_running(self):
        return self.running

    def set_request_interval(self, interval):
        self.interval = interval
        print(f"⏱️ Simulation request interval set to {interval} seconds.")

    def _get_current_hour(self):
        """
        Get the current simulated hour based on the clock.

        Returns:
            str: Hour range string (e.g., '08-09')
        """
        # Get simulated time from the clock
        sim_time = self.system.clock.get_time()

        # Convert to hours (assuming speed_factor is 10)
        current_hour = int((sim_time / 3600) % 24)
        next_hour = (current_hour + 1) % 24

        # Format as 'HH-HH'
        return f"{current_hour:02d}-{next_hour:02d}"

    def _generate_requests_for_hour(self, hour_key):
        """
        Generate requests for a specific hour based on data.

        Args:
            hour_key (str): Hour range key (e.g., '08-09')

        Returns:
            list: List of generated request tuples
        """
        graph = self.system.hospital.get_graph()
        locations = list(graph.get_nodes())

        # Default to 1 if no data found
        # Divide total requests by 365 to get daily average
        total_hourly_requests = self.hourly_request_data.get(hour_key, {}).get('request_count', 365)
        daily_hourly_rate = total_hourly_requests / 365

        # Calculate number of requests to generate
        num_requests = max(1, int(daily_hourly_rate * (self.interval / 3600)))

        requests = []
        od_pairs = self.od_pairs_by_hour.get(hour_key, [])

        for _ in range(num_requests):
            if od_pairs:
                # Choose OD pair based on probabilities
                od_choice = random.choices(
                    od_pairs,
                    weights=[pair.get('probability', 1) for pair in od_pairs],
                    k=1
                )[0]
                origin = od_choice['origin']
                destination = od_choice['destination']
            else:
                # Fallback to random selection
                origin, destination = random.sample(locations, 2)

            # Random urgency and transport type
            urgent = random.random() < 0.2  # 20% chance of being urgent
            transport_type = random.choice(["stretcher", "wheelchair", "bed"])

            requests.append((origin, destination, transport_type, urgent))

        return requests

    def _run_loop(self):
        """Main simulation loop."""
        while self.running:
            # Get current simulated hour
            current_hour = self._get_current_hour()

            # Generate requests for this hour
            requests = self._generate_requests_for_hour(current_hour)

            # Process each request
            for origin, destination, transport_type, urgent in requests:
                request = self.system.create_transport_request(origin, destination, transport_type, urgent)

                self.socketio.emit("simulation_event", {
                    "type": "new_request",
                    "origin": origin,
                    "destination": destination,
                    "transport_type": transport_type,
                    "urgent": urgent
                })
                self.system.log_event(
                    f"🆕 New request created: {origin} ➝ {destination} ({transport_type}, urgent={urgent})"
                )

                print(f"🧪 [Simulation] Request: {origin} ➝ {destination} ({transport_type}, urgent={urgent})")

                self.system.transport_manager.deploy_strategy_assignment()

            # Sleep for the interval
            eventlet.sleep(self.interval)