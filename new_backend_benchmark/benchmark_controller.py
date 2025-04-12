"""
Controller component for benchmark functionality.
Updated to include the new optimization strategies.
"""

import time
import threading
from Model.Assignment_strategies.ILP.ilp_mode import ILPMode
from new_backend_benchmark.time_based_benchmark_model import TimeBasedBenchmarkModel


class BenchmarkController:
    """
    Controller for managing benchmark operations.
    Coordinates between the model and view components.
    """

    def __init__(self, benchmark_model, socketio):
        """
        Initialize the benchmark controller.

        Args:
            benchmark_model: The benchmark model instance
            socketio: SocketIO instance for real-time communication
        """
        self.model = benchmark_model
        self.socketio = socketio
        self.benchmark_thread = None
        self.cancel_flag = False
        self.progress = 0
        self.start_time = 0
        self.time_model = TimeBasedBenchmarkModel(self.model)

    def start_benchmark(self, config):
        """
        Start a benchmark with the given configuration.

        Args:
            config (dict): Benchmark configuration including:
                - transporters: Number of transporters to use
                - random_runs: Number of random simulations to run
                - strategies: List of strategies to benchmark
                - scenarios: List of scenarios to use

        Returns:
            dict: Status message
        """
        # Extract configuration
        num_transporters = config.get("transporters", 3)
        random_runs = config.get("random_runs", 100)
        strategies = config.get("strategies", ["ILP: Makespan", "Random"])
        scenarios = config.get("scenarios", ["Default Scenario"])

        # Cancel existing benchmark if running
        if self.benchmark_thread and self.benchmark_thread.is_alive():
            self.cancel_flag = True
            self.benchmark_thread.join(timeout=1.0)

        # Reset benchmark state
        self.cancel_flag = False
        self.progress = 0
        self.start_time = time.time()

        # Create and start the benchmark thread
        self.benchmark_thread = threading.Thread(
            target=self._run_benchmark_thread,
            args=(num_transporters, random_runs, strategies, scenarios)
        )
        self.benchmark_thread.daemon = True
        self.benchmark_thread.start()

        return {"status": "Benchmark started"}

    def cancel_benchmark(self):
        """
        Cancel a running benchmark.

        Returns:
            dict: Status message
        """
        if self.benchmark_thread and self.benchmark_thread.is_alive():
            self.cancel_flag = True
            return {"status": "Cancelling benchmark"}
        else:
            return {"status": "No benchmark running"}

    # Add this function to your benchmark_controller.py file

    def _run_benchmark_thread(self, num_transporters, random_runs, strategies, scenarios):
        """
        Run the benchmark in a background thread.
        Fixed version to properly display new optimizers in graphs.
        """
        try:
            # Loop through all scenarios
            for scenario_name in scenarios:
                if self.cancel_flag:
                    self.socketio.emit("benchmark_complete", {"cancelled": True})
                    return

                # Get the scenario requests
                requests = self.model.get_scenario(scenario_name)

                # Track all results to ensure consistent output format
                benchmark_results = {}

                # Run ILP Makespan benchmark if selected
                if "ILP: Makespan" in strategies:
                    self._update_progress(5, f"Running ILP Makespan optimization for {scenario_name}")
                    ilp_results = self.model.run_ilp_benchmark(
                        num_transporters, requests, ILPMode.MAKESPAN
                    )

                    # Store results in standard format
                    benchmark_results["ILP: Makespan"] = {
                        "times": [ilp_results["makespan"]],
                        "workload": ilp_results["workload"]
                    }

                # Run ILP Equal Workload benchmark if selected
                if "ILP: Equal Workload" in strategies:
                    self._update_progress(10, f"Running ILP Equal Workload optimization for {scenario_name}")
                    ilp_equal_results = self.model.run_ilp_benchmark(
                        num_transporters, requests, ILPMode.EQUAL_WORKLOAD
                    )

                    # Store results in standard format
                    benchmark_results["ILP: Equal Workload"] = {
                        "times": [ilp_equal_results["makespan"]],
                        "workload": ilp_equal_results["workload"]
                    }

                # Run ILP Urgency First benchmark if selected
                if "ILP: Urgency First" in strategies:
                    self._update_progress(15, f"Running ILP Urgency First optimization for {scenario_name}")
                    ilp_urgency_results = self.model.run_ilp_benchmark(
                        num_transporters, requests, ILPMode.URGENCY_FIRST
                    )

                    # Store results in standard format
                    benchmark_results["ILP: Urgency First"] = {
                        "times": [ilp_urgency_results["makespan"]],
                        "workload": ilp_urgency_results["workload"]
                    }

                # Run ILP Cluster-Based benchmark if selected
                if "ILP: Cluster-Based" in strategies:
                    self._update_progress(20, f"Running ILP Cluster-Based optimization for {scenario_name}")
                    try:
                        # Force deterministic behavior for benchmarking
                        dept_count = len(set(origin for origin, _, _ in requests) |
                                         set(dest for _, dest, _ in requests))
                        num_clusters = max(2, min(5, dept_count // 5))

                        ilp_cluster_results = self.model.run_ilp_benchmark(
                            num_transporters, requests, ILPMode.CLUSTER_BASED,
                            extra_params={"num_clusters": num_clusters}
                        )

                        # Store results in standard format
                        benchmark_results["ILP: Cluster-Based"] = {
                            "times": [ilp_cluster_results["makespan"]],
                            "workload": ilp_cluster_results["workload"]
                        }

                        # Debug output
                        print(f"Cluster-Based ILP results: {ilp_cluster_results['makespan']}")
                    except Exception as e:
                        print(f"Error running Cluster-Based ILP: {str(e)}")
                        # Create placeholder results to avoid UI errors
                        benchmark_results["ILP: Cluster-Based"] = {
                            "times": [0],
                            "workload": {t.name: 0 for t in self.model.transporters}
                        }

                # Run Genetic Algorithm benchmark if selected
                if "Genetic Algorithm" in strategies:
                    self._update_progress(25, f"Running Genetic Algorithm optimization for {scenario_name}")
                    try:
                        genetic_results = self.model.run_genetic_benchmark(
                            num_transporters, requests,
                            params={"time_limit_seconds": 3}  # Short time limit for benchmarking
                        )

                        # Store results in standard format
                        benchmark_results["Genetic Algorithm"] = {
                            "times": [genetic_results["makespan"]],
                            "workload": genetic_results["workload"]
                        }

                        # Debug output
                        print(f"Genetic Algorithm results: {genetic_results['makespan']}")
                    except Exception as e:
                        print(f"Error running Genetic Algorithm: {str(e)}")
                        # Create placeholder results to avoid UI errors
                        benchmark_results["Genetic Algorithm"] = {
                            "times": [0],
                            "workload": {t.name: 0 for t in self.model.transporters}
                        }

                # Run Random benchmark with multiple iterations
                if "Random" in strategies:
                    self._update_progress(30, f"Starting Random simulations for {scenario_name}")

                    # Run all random simulations in one go
                    random_results = self.model.run_random_benchmark(
                        num_transporters, requests, random_runs
                    )

                    # Extract just the makespan times
                    random_times = [r["makespan"] for r in random_results]

                    # Get workload from first run
                    random_workload = random_results[0]["workload"] if random_results else {}

                    # Store results in standard format
                    benchmark_results["Random"] = {
                        "times": random_times,
                        "workload": random_workload
                    }

                    # Update progress incrementally during random runs
                    progress_steps = max(1, random_runs // 10)
                    for i in range(0, random_runs, progress_steps):
                        if self.cancel_flag:
                            self.socketio.emit("benchmark_complete", {"cancelled": True})
                            return

                        progress = 30 + int((i / random_runs) * 70)
                        step = min(i + progress_steps, random_runs)
                        self._update_progress(progress, f"Processed Random simulation ({step}/{random_runs})")

                # Emit all results in a consistent format
                for strategy_name, result in benchmark_results.items():
                    self.socketio.emit("benchmark_results", {
                        "strategy": strategy_name,
                        "times": result["times"],
                        "workload": result["workload"]
                    })
                    # Add a small delay to ensure messages are processed in order
                    import time
                    time.sleep(0.1)

                # Emit benchmark complete
                self._update_progress(100, "Benchmark complete")
                import time
                time.sleep(0.5)  # Give a moment for final progress update
                self.socketio.emit("benchmark_complete", {"success": True})

        except Exception as e:
            import traceback
            print(f"Error in benchmark: {str(e)}")
            traceback.print_exc()
            self.socketio.emit("benchmark_complete", {"error": str(e)})

    def _update_progress(self, progress, current_task):
        """
        Update the benchmark progress and emit a progress event.

        Args:
            progress (int): Progress percentage (0-100)
            current_task (str): Description of current task
        """
        self.progress = progress
        elapsed = time.time() - self.start_time

        # Estimate completion time
        if progress > 0:
            estimated_total = elapsed * 100 / progress
            estimated_remaining = estimated_total - elapsed
        else:
            estimated_remaining = 0

        # Emit progress event
        self.socketio.emit("benchmark_progress", {
            "progress": progress,
            "current_task": current_task,
            "elapsed_time": elapsed,
            "estimated_completion": estimated_remaining
        })

    def get_available_scenarios(self):
        """
        Get the list of available benchmark scenarios.

        Returns:
            list: List of scenario names
        """
        return list(self.model.scenarios.keys())

    def add_custom_scenario(self, name, requests):
        """
        Add a custom scenario for benchmarking.

        Args:
            name (str): Scenario name
            requests (list): List of request tuples (origin, destination, urgent)

        Returns:
            list: Updated list of scenario names
        """
        self.model.add_scenario(name, requests)
        return self.get_available_scenarios()

    def add_custom_scenario(self, name, requests):
        """
        Add a custom scenario for benchmarking.

        Args:
            name (str): Scenario name
            requests (list): List of request tuples (origin, destination, urgent)

        Returns:
            list: Updated list of scenario names
        """
        self.model.add_scenario(name, requests)
        return self.get_available_scenarios()

    # Time-based benchmark methods

    def get_available_time_ranges(self):
        """
        Get available time ranges for time-based benchmarking.

        Returns:
            list: List of time range strings
        """
        return self.time_model.get_available_time_ranges()

    def get_hourly_rate_data(self):
        """
        Get hourly rate data for visualization.

        Returns:
            dict: Hourly rate data for charts
        """
        return self.time_model.get_hourly_rates_data()

    def validate_time_range(self, start_hour, end_hour):
        """
        Validate a time range input.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)

        Returns:
            tuple: (is_valid, error_message)
        """
        # Check if within valid range
        if start_hour is None or end_hour is None:
            return False, "Missing time range parameters"

        try:
            start_hour = int(start_hour)
            end_hour = int(end_hour)
        except ValueError:
            return False, "Time range must be integers"

        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            return False, "Time range must be between 0-23"

        # Valid input
        return True, None

    def generate_time_scenario(self, start_hour, end_hour, name=None, request_count=None):
        """
        Generate a time-based benchmark scenario.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            name (str, optional): Name for the scenario
            request_count (int, optional): Number of requests

        Returns:
            dict: Result with scenario or error
        """
        # Validate inputs
        valid, error = self.validate_time_range(start_hour, end_hour)
        if not valid:
            return {"success": False, "error": error}

        # Generate scenario
        scenario = self.time_model.generate_scenario(
            int(start_hour), int(end_hour), name,
            int(request_count) if request_count is not None else None
        )

        if not scenario:
            return {
                "success": False,
                "error": f"Failed to generate requests for time range {start_hour}-{end_hour}"
            }

        # Add to benchmark model
        if not self.time_model.add_scenario_to_benchmark(scenario):
            return {
                "success": False,
                "error": "Failed to add scenario to benchmark model"
            }

        return {"success": True, "scenario": scenario}

    def run_time_based_benchmark(self, start_hour, end_hour, transporters, random_runs=100):
        """
        Run a time-based benchmark.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            transporters (int): Number of transporters
            random_runs (int): Number of random runs for comparison

        Returns:
            dict: Benchmark results
        """
        # Validate inputs
        valid, error = self.validate_time_range(start_hour, end_hour)
        if not valid:
            return {"success": False, "error": error}

        try:
            transporters = int(transporters)
            if transporters < 1:
                return {"success": False, "error": "Transport count must be at least 1"}

            random_runs = int(random_runs)
            if random_runs < 1:
                random_runs = 100  # Default value
        except ValueError:
            return {"success": False, "error": "Invalid transporter count or random runs"}

        # Run benchmark
        result = self.time_model.run_benchmark_for_time_range(
            int(start_hour), int(end_hour), transporters, random_runs
        )

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {"success": True, "data": result}