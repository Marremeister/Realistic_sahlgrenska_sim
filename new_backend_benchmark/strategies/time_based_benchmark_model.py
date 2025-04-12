"""
Model component for time-based benchmark functionality.
Handles data and core functionality for time-based benchmarks.
"""
import os
import json
import logging
from new_backend_benchmark.execution.repository.transport_data_repository import TransportDataRepository


class TimeBasedBenchmarkModel:
    """
    Model for time-based benchmarking functionality.
    Handles data loading, request generation, and benchmark data preparation.
    """

    def __init__(self, benchmark_model, data_dir='analysis_output'):
        """
        Initialize the time-based benchmark model.

        Args:
            benchmark_model: Base benchmark model to use for running benchmarks
            data_dir: Directory containing time-based analysis data
        """
        self.benchmark_model = benchmark_model
        self.data_dir = data_dir
        self.repository = TransportDataRepository(data_dir)
        self.logger = logging.getLogger("TimeBasedBenchmarkModel")

    def get_available_time_ranges(self):
        """
        Get available time ranges and common business hours.

        Returns:
            list: List of time range strings
        """
        # Get time ranges from repository
        time_ranges = self.repository.get_available_time_ranges()

        # Add common business hours if not already included
        common_ranges = ["08-17", "09-17", "08-12", "13-17"]
        for common_range in common_ranges:
            if common_range not in time_ranges:
                time_ranges.append(common_range)

        return sorted(time_ranges)

    def get_hourly_rates_data(self):
        """
        Get hourly request rate data for visualization.

        Returns:
            dict: Hourly rate data suitable for charts
        """
        return self.repository.get_hourly_rates_for_chart()

    def generate_requests_for_time_range(self, start_hour, end_hour, request_count=None, debug_export=True):
        """Generate realistic transport requests for the specified time range."""
        # If request count is provided, use that
        if request_count is not None:
            return self.repository.generate_benchmark_requests(start_hour, end_hour, request_count)

        # Otherwise, get rate and calculate a daily amount
        hourly_rate = self.repository.get_request_rate(start_hour, end_hour)
        hours = end_hour - start_hour if end_hour > start_hour else (24 - start_hour) + end_hour

        # Apply daily scaling - divide by approximate days in dataset (e.g., 250 work days in year)
        # This transforms the yearly total into a daily average
        estimated_days_in_dataset = 365  # Adjust based on your dataset
        daily_rate = hourly_rate / estimated_days_in_dataset

        # Calculate reasonable number of requests for one day
        num_requests = int(daily_rate * hours)
        num_requests = max(1, min(num_requests, 200))  # Reasonable bounds

        requests = self.repository.generate_benchmark_requests(start_hour, end_hour, num_requests)

        # Export for debugging if requested

        if debug_export:
            self._export_debug_requests(requests, start_hour, end_hour)

        return requests

    def generate_scenario(self, start_hour, end_hour, name=None, request_count=None):
        """
        Generate a complete benchmark scenario for the specified time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            name (str, optional): Name for the scenario (if None, generated based on time range)
            request_count (int, optional): Number of requests (if None, calculated based on data)

        Returns:
            dict: Scenario data including requests and metadata
        """
        # Generate default name if not provided
        if not name:
            # Morning: 5-12, Afternoon: 12-17, Evening: 17-21, Night: 21-5
            if 5 <= start_hour < 12:
                period = "Morning"
            elif 12 <= start_hour < 17:
                period = "Afternoon"
            elif 17 <= start_hour < 21:
                period = "Evening"
            else:
                period = "Night"

            name = f"{period} {start_hour:02d}-{end_hour:02d}"

        # Generate requests
        requests = self.generate_requests_for_time_range(start_hour, end_hour, request_count)

        if not requests:
            return None

        # Count urgent requests
        urgent_count = sum(1 for _, _, _, urgent in requests if urgent)

        # Convert to scenario format
        scenario_requests = []
        for origin, dest, transport_type, urgent in requests:
            scenario_requests.append({
                "origin": origin,
                "destination": dest,
                "transport_type": transport_type,
                "urgent": urgent
            })

        # Create full scenario
        scenario = {
            "name": name,
            "time_range": f"{start_hour:02d}-{end_hour:02d}",
            "requests": scenario_requests,
            "urgent_count": urgent_count,
            "request_count": len(requests),
            "hourly_rate": self.repository.get_request_rate(start_hour, end_hour)
        }

        return scenario

    def add_scenario_to_benchmark(self, scenario):
        """
        Add a scenario to the benchmark model for testing.

        Args:
            scenario (dict): Scenario data

        Returns:
            bool: True if added successfully
        """
        try:
            # Extract request tuples
            requests = [
                (req["origin"], req["destination"], req["urgent"])
                for req in scenario["requests"]
            ]

            # Add to benchmark model
            self.benchmark_model.add_scenario(scenario["name"], requests)
            return True
        except Exception as e:
            self.logger.error(f"Error adding scenario to benchmark: {str(e)}")
            return False

    def run_benchmark_for_time_range(self, start_hour, end_hour, transporter_count, strategies=None, random_runs=100):
        """
        Generate a scenario for a specific time range, but don't run the benchmark directly.
        This preserves modularity with the existing strategy selection system.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            transporter_count (int): Number of transporters
            strategies (list): Optimization strategies (handled by main benchmark system)
            random_runs (int): Number of random runs for comparison

        Returns:
            dict: Scenario data for benchmarking
        """
        try:
            # Generate scenario
            scenario = self.generate_scenario(start_hour, end_hour)
            if not scenario:
                return {"error": "Failed to generate scenario"}

            # Add to benchmark model
            if not self.add_scenario_to_benchmark(scenario):
                return {"error": "Failed to add scenario to benchmark model"}

            # Return the scenario info - the main benchmark system will handle running the benchmark
            return {
                "scenario": scenario,
                "scenario_name": scenario["name"]
            }

        except Exception as e:
            self.logger.error(f"Error preparing time-based benchmark: {str(e)}")
            return {"error": str(e)}

    def _export_debug_requests(self, requests, start_hour, end_hour):
        """
        Export requests to a JSON file for debugging.

        Args:
            requests: List of (origin, destination, transport_type, urgent) tuples
            start_hour: Start hour
            end_hour: End hour
        """

        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(self.data_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        # Convert requests to a more readable format
        request_data = [
            {
                "origin": origin,
                "destination": dest,
                "transport_type": t_type,
                "urgent": urgent
            }
            for origin, dest, t_type, urgent in requests
        ]

        # Extract unique department names
        departments = set()
        for req in request_data:
            departments.add(req["origin"])
            departments.add(req["destination"])

        # Create complete export data
        export_data = {
            "time_range": f"{start_hour:02d}-{end_hour:02d}",
            "requests": request_data,
            "unique_departments": sorted(list(departments)),
            "count": len(request_data)
        }

        # Write to file
        output_file = os.path.join(debug_dir, f'requests_{start_hour:02d}-{end_hour:02d}.json')
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        # Also save graph nodes for comparison
        try:
            # Check if we have a saved hospital graph
            graph_file = os.path.join(self.data_dir, 'hospital_graph.json')
            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    graph_data = json.load(f)

                graph_depts = set(graph_data.get("departments", []))

                # Compare departments
                matching = departments.intersection(graph_depts)
                missing = departments - graph_depts

                comparison = {
                    "matching_count": len(matching),
                    "missing_count": len(missing),
                    "missing_departments": sorted(list(missing))
                }

                # Write comparison to file
                comparison_file = os.path.join(debug_dir, f'comparison_{start_hour:02d}-{end_hour:02d}.json')
                with open(comparison_file, 'w') as f:
                    json.dump(comparison, f, indent=2)

                if missing:
                    self.logger.warning(
                        f"Found {len(missing)} departments in requests that are not in the hospital graph!")
                    for dept in sorted(missing):
                        self.logger.warning(f"  - Missing department: {dept}")
        except Exception as e:
            self.logger.error(f"Error comparing departments: {str(e)}")

        self.logger.info(
            f"Exported {len(request_data)} requests with {len(departments)} unique departments to {output_file}")