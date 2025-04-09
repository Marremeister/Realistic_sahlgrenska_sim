import os
import json
import logging
from datetime import datetime


class TimeBasedBenchmarkGenerator:
    """
    Generator for time-based benchmark scenarios using pre-analyzed transport data.
    Integrates with the BenchmarkModel to create realistic benchmark scenarios.
    """

    def __init__(self, data_repository):
        """
        Initialize the generator with a data repository.

        Args:
            data_repository: TransportDataRepository instance with loaded data
        """
        self.repository = data_repository
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up a logger for the TimeBasedBenchmarkGenerator."""
        logger = logging.getLogger("TimeBasedBenchmarkGenerator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def generate_benchmark_scenario(self, start_hour, end_hour, num_transporters, num_requests=None):
        """
        Generate a complete benchmark scenario for the specified time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            num_transporters (int): Number of transporters to use
            num_requests (int, optional): Number of requests to generate.
                                         If None, calculated based on average rate.

        Returns:
            dict: Benchmark scenario configuration
        """
        # Generate requests
        requests = self.repository.generate_benchmark_requests(start_hour, end_hour, num_requests)

        # Create transporter names
        transporters = [f"T{i + 1}" for i in range(num_transporters)]

        # Create scenario
        scenario = {
            "name": f"Time-based {start_hour:02d}-{end_hour:02d}",
            "description": f"Generated scenario for {start_hour:02d}:00-{end_hour:02d}:00",
            "transporters": transporters,
            "requests": [
                {
                    "origin": origin,
                    "destination": dest,
                    "transport_type": t_type,
                    "urgent": urgent
                }
                for origin, dest, t_type, urgent in requests
            ],
            "metadata": {
                "time_range": f"{start_hour:02d}-{end_hour:02d}",
                "generated_at": datetime.now().isoformat(),
                "hourly_rate": self.repository.get_request_rate(start_hour, end_hour),
                "num_transporters": num_transporters,
                "num_requests": len(requests)
            }
        }

        return scenario

    def save_benchmark_scenario(self, scenario, output_dir='benchmark_scenarios'):
        """
        Save a benchmark scenario to a file.

        Args:
            scenario (dict): Benchmark scenario configuration
            output_dir (str): Directory to save the scenario

        Returns:
            str: Path to the saved scenario file
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create filename
        time_range = scenario['metadata']['time_range']
        num_transporters = scenario['metadata']['num_transporters']
        num_requests = scenario['metadata']['num_requests']

        filename = f"scenario_{time_range}_{num_transporters}t_{num_requests}r.json"
        filepath = os.path.join(output_dir, filename)

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(scenario, f, indent=2)

        self.logger.info(f"Saved benchmark scenario to {filepath}")
        return filepath

    def load_benchmark_scenario(self, filepath):
        """
        Load a benchmark scenario from a file.

        Args:
            filepath (str): Path to the scenario file

        Returns:
            dict: Benchmark scenario configuration
        """
        try:
            with open(filepath, 'r') as f:
                scenario = json.load(f)

            self.logger.info(f"Loaded benchmark scenario from {filepath}")
            return scenario
        except Exception as e:
            self.logger.error(f"Error loading benchmark scenario: {str(e)}")
            return None

    def get_available_scenarios(self, scenario_dir='benchmark_scenarios'):
        """
        Get a list of available benchmark scenarios.

        Args:
            scenario_dir (str): Directory containing scenario files

        Returns:
            list: List of scenario metadata
        """
        if not os.path.exists(scenario_dir):
            return []

        scenarios = []

        for filename in os.listdir(scenario_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(scenario_dir, filename)
                    with open(filepath, 'r') as f:
                        scenario = json.load(f)

                    # Extract key metadata
                    scenarios.append({
                        "name": scenario.get("name", filename),
                        "time_range": scenario.get("metadata", {}).get("time_range", "unknown"),
                        "num_transporters": scenario.get("metadata", {}).get("num_transporters", 0),
                        "num_requests": scenario.get("metadata", {}).get("num_requests", 0),
                        "filepath": filepath
                    })
                except Exception as e:
                    self.logger.warning(f"Error loading scenario {filename}: {str(e)}")

        return scenarios

    def convert_scenario_to_benchmark_format(self, scenario):
        """
        Convert a scenario to the format expected by the benchmark model.

        Args:
            scenario (dict): Benchmark scenario configuration

        Returns:
            tuple: (transporters, requests) in benchmark model format
        """
        transporters = scenario["transporters"]

        # Convert requests to (origin, destination) tuples
        requests = [
            (req["origin"], req["destination"])
            for req in scenario["requests"]
        ]

        return transporters, requests

    def benchmark_from_time_range(self, benchmark_model, start_hour, end_hour,
                                  num_transporters, random_runs=1000, save_scenario=True):
        """
        Run a benchmark directly from time range parameters.

        Args:
            benchmark_model: BenchmarkModel instance
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            num_transporters (int): Number of transporters to use
            random_runs (int): Number of random runs to perform
            save_scenario (bool): Whether to save the generated scenario

        Returns:
            tuple: (optimal_times, random_times, workload_opt, workload_rand)
        """
        # Generate scenario
        scenario = self.generate_benchmark_scenario(start_hour, end_hour, num_transporters)

        # Save scenario if requested
        if save_scenario:
            self.save_benchmark_scenario(scenario)

        # Convert to benchmark format
        transporters, requests = self.convert_scenario_to_benchmark_format(scenario)

        # Run benchmarks
        optimal_times = benchmark_model.run_benchmark("ilp", 1, transporters, requests)
        random_times = benchmark_model.run_benchmark("random", random_runs, transporters, requests)

        # Get workload distributions
        workload_opt = benchmark_model.get_workload_distribution("ilp", transporters, requests)
        workload_rand = benchmark_model.get_workload_distribution("random", transporters, requests)

        return optimal_times, random_times, workload_opt, workload_rand


# Integrate with the benchmark controller
class TimeBasedBenchmarkController:
    """
    Controller for time-based benchmarking, integrating with the existing
    benchmark infrastructure.
    """

    def __init__(self, benchmark_controller, data_dir='analysis_output'):
        """
        Initialize the controller.

        Args:
            benchmark_controller: Existing BenchmarkController instance
            data_dir (str): Directory containing analysis data
        """
        from new_backend_benchmark.transport_data_repository import TransportDataRepository

        self.benchmark_controller = benchmark_controller
        self.model = benchmark_controller.model

        # Initialize repository and generator
        self.repository = TransportDataRepository(data_dir)
        self.generator = TimeBasedBenchmarkGenerator(self.repository)

        self.logger = logging.getLogger("TimeBasedBenchmarkController")

    def run_time_based_benchmark(self, time_range, num_transporters, random_runs=1000):
        """
        Run a time-based benchmark.

        Args:
            time_range (str): Time range in format "HH-HH" (e.g., "08-17")
            num_transporters (int): Number of transporters to use
            random_runs (int): Number of random runs to perform

        Returns:
            dict: Benchmark results
        """
        try:
            # Parse time range
            start_hour, end_hour = map(int, time_range.split('-'))

            # Generate scenario and run benchmark
            optimal_times, random_times, workload_opt, workload_rand = self.generator.benchmark_from_time_range(
                self.model, start_hour, end_hour, num_transporters, random_runs
            )

            # Calculate statistics similar to the original benchmark controller
            import numpy as np

            opt_std = self.model.calculate_workload_std(workload_opt)
            rand_std = self.model.calculate_workload_std(workload_rand)

            # Similar output as the original run_and_plot but without plotting
            self.logger.info(f"\n🔬 Time-based benchmark {time_range} with {num_transporters} transporters")
            self.logger.info(f"✅ Optimal Time: {optimal_times[0]:.2f} sec")
            self.logger.info(f"🎲 Random Avg:  {np.mean(random_times):.2f} sec")

            # Return results
            return {
                "scenario": {
                    "time_range": time_range,
                    "num_transporters": num_transporters,
                    "random_runs": random_runs
                },
                "results": {
                    "optimal_time": optimal_times[0],
                    "random_times": random_times,
                    "random_mean": float(np.mean(random_times)),
                    "random_std": float(np.std(random_times)),
                    "random_median": float(np.median(random_times)),
                    "improvement_percentage": float(
                        ((np.mean(random_times) - optimal_times[0]) / np.mean(random_times)) * 100)
                },
                "workload": {
                    "optimal": workload_opt,
                    "random": workload_rand,
                    "optimal_std": float(opt_std),
                    "random_std": float(rand_std)
                }
            }

        except Exception as e:
            self.logger.error(f"Error running time-based benchmark: {str(e)}")
            return {"error": str(e)}

    def get_available_time_ranges(self):
        """
        Get all available time ranges for benchmarking.

        Returns:
            list: List of available time range strings
        """
        # Return predefined time ranges from repository
        predefined = self.repository.get_available_time_ranges()

        # Add common time ranges if not already in predefined
        common_ranges = ["08-17", "09-17", "08-12", "13-17"]

        return sorted(list(set(predefined + common_ranges)))

    def get_hourly_rate_data(self):
        """
        Get hourly rate data for charts.

        Returns:
            dict: Hourly rate data suitable for charts
        """
        return self.repository.get_hourly_rates_for_chart()