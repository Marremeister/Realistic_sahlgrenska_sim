"""
Controller component for benchmark functionality.
Coordinates between the model and view, handling user interactions.
"""

import time
import threading
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple


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
        self.logger = self._setup_logger()
        self.time_model = None

    def _setup_logger(self):
        """Set up a logger for the controller."""
        logger = logging.getLogger("BenchmarkController")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def start_benchmark(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Start a benchmark with the given configuration.

        Args:
            config: Benchmark configuration including:
                - transporters: Number of transporters to use
                - random_runs: Number of random simulations to run
                - strategies: List of strategies to benchmark
                - scenarios: List of scenarios to use
                - incremental_mode: Whether to run in incremental mode
                - time_range: Optional time range for incremental mode
                - time_distribution: How to distribute requests in time

        Returns:
            Dict: Status message
        """
        try:
            # Extract and validate configuration
            config = self._validate_benchmark_config(config)

            # Cancel existing benchmark if running
            self._cancel_existing_benchmark()

            # Reset benchmark state
            self._reset_benchmark_state()

            # Create and start the benchmark thread
            self._start_benchmark_thread(config)

            return {"status": "Benchmark started"}
        except Exception as e:
            self.logger.error(f"Error starting benchmark: {str(e)}")
            return {"status": "Error", "message": str(e)}

    def _validate_benchmark_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and extract benchmark configuration.

        Args:
            config: Raw configuration dictionary

        Returns:
            Dict: Validated configuration with defaults applied
        """
        # Apply defaults for missing values
        validated = self._apply_config_defaults(config)

        # Validate transporters
        self._validate_transporter_count(validated)

        # Validate random runs
        self._validate_random_runs(validated)

        # Validate strategies
        self._validate_strategies(validated)

        # Validate scenarios
        self._validate_scenarios(validated)

        # Validate incremental mode settings
        self._validate_incremental_settings(validated)

        return validated

    def _apply_config_defaults(self, config):
        """Apply default values to missing configuration options."""
        return {
            "transporters": config.get("transporters", 3),
            "random_runs": config.get("random_runs", 100),
            "strategies": config.get("strategies", ["ILP: Makespan", "Random"]),
            "scenarios": config.get("scenarios", ["Default Scenario"]),
            "incremental_mode": config.get("incremental_mode", False),
            "time_range": config.get("time_range", [0, 3600]),  # Default 1 hour
            "time_distribution": config.get("time_distribution", "realistic")
        }

    def _validate_transporter_count(self, config):
        """Validate the number of transporters."""
        if config["transporters"] <= 0:
            raise ValueError("Number of transporters must be greater than 0")

    def _validate_random_runs(self, config):
        """Validate the number of random runs."""
        if config["random_runs"] <= 0:
            config["random_runs"] = 100

    def _validate_strategies(self, config):
        """Validate the selected strategies."""
        available_strategies = [s["name"] for s in self.model.get_available_strategies()]
        config["strategies"] = [s for s in config["strategies"] if s in available_strategies]
        if not config["strategies"]:
            config["strategies"] = ["ILP: Makespan", "Random"]

    def _validate_scenarios(self, config):
        """Validate the selected scenarios."""
        available_scenarios = self.model.get_available_scenarios()
        config["scenarios"] = [s for s in config["scenarios"] if s in available_scenarios]
        if not config["scenarios"]:
            config["scenarios"] = ["Default Scenario"]

    def _validate_incremental_settings(self, config):
        """Validate incremental mode settings."""
        if config["incremental_mode"]:
            time_range = config["time_range"]
            # Check if time range is valid
            if not isinstance(time_range, list) or len(time_range) != 2:
                config["time_range"] = [0, 3600]  # Default 1 hour
            elif time_range[0] >= time_range[1]:
                config["time_range"] = [0, 3600]  # Default 1 hour

            # Check if time distribution is valid
            valid_distributions = ["random", "uniform", "realistic"]
            if config["time_distribution"] not in valid_distributions:
                config["time_distribution"] = "realistic"

    def _cancel_existing_benchmark(self):
        """Cancel any existing benchmark thread."""
        if self.benchmark_thread and self.benchmark_thread.is_alive():
            self.cancel_flag = True
            self.benchmark_thread.join(timeout=1.0)

    def _reset_benchmark_state(self):
        """Reset the benchmark state for a new run."""
        self.cancel_flag = False
        self.progress = 0
        self.start_time = time.time()

    def _start_benchmark_thread(self, config: Dict[str, Any]):
        """
        Start the benchmark thread.

        Args:
            config: Validated benchmark configuration
        """
        self.benchmark_thread = threading.Thread(
            target=self._run_benchmark_thread,
            args=(
                config["transporters"],
                config["random_runs"],
                config["strategies"],
                config["scenarios"],
                config["incremental_mode"],
                config["time_range"],
                config["time_distribution"]
            )
        )
        self.benchmark_thread.daemon = True
        self.benchmark_thread.start()

    def cancel_benchmark(self) -> Dict[str, str]:
        """
        Cancel a running benchmark.

        Returns:
            Dict: Status message
        """
        if self.benchmark_thread and self.benchmark_thread.is_alive():
            self.cancel_flag = True
            return {"status": "Cancelling benchmark"}
        else:
            return {"status": "No benchmark running"}

    def _run_benchmark_thread(
        self, num_transporters: int, random_runs: int,
        strategy_names: List[str], scenario_names: List[str],
        incremental_mode: bool = False, time_range: List[float] = None,
        time_distribution: str = "realistic"
    ) -> None:
        """
        Run the benchmark in a background thread.

        Args:
            num_transporters: Number of transporters to use
            random_runs: Number of random simulations to run
            strategy_names: List of strategy names to benchmark
            scenario_names: List of scenario names to use
            incremental_mode: Whether to run in incremental mode
            time_range: Optional time range for incremental mode
            time_distribution: How to distribute requests in time
        """
        try:
            # Loop through all scenarios
            for scenario_name in scenario_names:
                if self._check_cancel_flag():
                    return

                # Get the scenario requests
                requests = self.model.get_scenario(scenario_name)

                # Track all results
                benchmark_results = {}

                # Run all strategies for this scenario
                self._run_all_strategies(
                    benchmark_results,
                    strategy_names,
                    num_transporters,
                    requests,
                    random_runs,
                    scenario_name,
                    incremental_mode,
                    time_range,
                    time_distribution
                )

                # Emit results to the client
                self._emit_results(benchmark_results)

                # Finalize the benchmark
                self._finalize_benchmark()

        except Exception as e:
            self._handle_benchmark_error(e)

    def _check_cancel_flag(self) -> bool:
        """
        Check if the benchmark should be cancelled.

        Returns:
            bool: True if cancelled, False otherwise
        """
        if self.cancel_flag:
            self.socketio.emit("benchmark_complete", {"cancelled": True})
            return True
        return False

    def _run_all_strategies(
        self, results: Dict[str, Any], strategy_names: List[str],
        num_transporters: int, requests: List[tuple],
        random_runs: int, scenario_name: str,
        incremental_mode: bool = False, time_range: List[float] = None,
        time_distribution: str = "realistic"
    ):
        """
        Run all requested strategies for a scenario.

        Args:
            results: Dictionary to store results
            strategy_names: List of strategy names to run
            num_transporters: Number of transporters to use
            requests: List of requests for the scenario
            random_runs: Number of random runs
            scenario_name: Name of the current scenario
            incremental_mode: Whether to run in incremental mode
            time_range: Optional time range for incremental mode
            time_distribution: How to distribute requests in time
        """
        # Calculate total steps for progress tracking
        total_strategies = len(strategy_names)

        # Run each strategy
        for i, strategy_name in enumerate(strategy_names):
            if self._check_cancel_flag():
                return

            # Calculate progress percentage for this strategy
            progress_base = int(i * 90 / total_strategies) + 5

            # Update progress
            run_mode = "incremental" if incremental_mode else "standard"
            self._update_progress(
                progress_base,
                f"Running {strategy_name} optimization ({run_mode} mode) for {scenario_name}"
            )

            # Run the appropriate strategy
            if strategy_name == "Random" and not incremental_mode:
                self._run_random_strategy(
                    results, num_transporters, requests, random_runs,
                    progress_base, i, total_strategies
                )
            else:
                # Run either standard or incremental benchmark
                self._run_single_strategy(
                    results, strategy_name, num_transporters, requests,
                    incremental_mode, time_range, time_distribution
                )

    def _run_random_strategy(
        self, results: Dict[str, Any], num_transporters: int,
        requests: List[tuple], random_runs: int,
        progress_base: int, strategy_index: int, total_strategies: int
    ):
        """
        Run the Random strategy with multiple iterations.

        Args:
            results: Dictionary to store results
            num_transporters: Number of transporters to use
            requests: List of requests
            random_runs: Number of random runs
            progress_base: Base progress percentage
            strategy_index: Index of this strategy
            total_strategies: Total number of strategies
        """
        try:
            # Run the random benchmark
            random_result = self.model.run_benchmark(
                "Random",
                num_transporters,
                requests,
                runs=random_runs
            )

            # Store results
            results["Random"] = {
                "times": random_result["times"],
                "workload": random_result["workload"]
            }

            # Update progress incrementally
            self._update_random_progress(random_runs, progress_base, strategy_index, total_strategies)

        except Exception as e:
            self.logger.error(f"Error in Random benchmark: {str(e)}")
            results["Random"] = {
                "times": [0],
                "workload": {},
                "error": str(e)
            }

    def _update_random_progress(
        self, random_runs: int, progress_base: int,
        strategy_index: int, total_strategies: int
    ):
        """
        Update progress during random simulations.

        Args:
            random_runs: Total number of random runs
            progress_base: Base progress percentage
            strategy_index: Index of this strategy
            total_strategies: Total number of strategies
        """
        progress_steps = max(1, random_runs // 10)
        progress_range = 90 // total_strategies

        for step in range(0, random_runs, progress_steps):
            if self._check_cancel_flag():
                return

            curr_progress = progress_base + int((step / random_runs) * progress_range)
            batch = min(step + progress_steps, random_runs)
            self._update_progress(
                curr_progress,
                f"Processed Random simulation ({batch}/{random_runs})"
            )

    def _run_single_strategy(
        self, results: Dict[str, Any], strategy_name: str,
        num_transporters: int, requests: List[tuple],
        incremental_mode: bool = False, time_range: List[float] = None,
        time_distribution: str = "realistic"
    ):
        """
        Run a single optimization strategy.

        Args:
            results: Dictionary to store results
            strategy_name: Name of the strategy to run
            num_transporters: Number of transporters to use
            requests: List of requests
            incremental_mode: Whether to run in incremental mode
            time_range: Optional time range for incremental mode
            time_distribution: How to distribute requests in time
        """
        try:
            # Prepare time range tuple if provided
            time_range_tuple = self._prepare_time_range(time_range, incremental_mode)

            # Run the benchmark
            result = self.model.run_benchmark(
                strategy_name,
                num_transporters,
                requests,
                incremental_mode=incremental_mode,
                time_range=time_range_tuple,
                time_distribution=time_distribution
            )

            # Store standard results
            results[strategy_name] = self._prepare_standard_results(result)

            # Store incremental results if applicable
            if incremental_mode:
                self._add_incremental_results(results[strategy_name], result)

                # Emit incremental results separately for real-time updates
                self._emit_incremental_results(strategy_name, result)

        except Exception as e:
            self._handle_strategy_error(results, strategy_name, e)

    def _prepare_time_range(self, time_range, incremental_mode):
        """Prepare time range tuple for incremental benchmarks."""
        if not incremental_mode or not time_range:
            return None
        return (time_range[0], time_range[1])

    def _prepare_standard_results(self, result):
        """Prepare standard result structure."""
        return {
            "times": [result["makespan"]],
            "workload": result["workload"],
            "incremental": result.get("incremental", False)
        }

    def _add_incremental_results(self, strategy_results, result):
        """Add incremental-specific results to the strategy results."""
        incremental_fields = [
            "time_metrics", "events", "simulation_time", "hourly_distribution"
        ]

        for field in incremental_fields:
            if field in result:
                strategy_results[field] = result[field]

    def _handle_strategy_error(self, results, strategy_name, error):
        """Handle an error during strategy execution."""
        self.logger.error(f"Error in {strategy_name} benchmark: {str(error)}")
        results[strategy_name] = {
            "times": [0],
            "workload": {},
            "error": str(error)
        }

    def _emit_incremental_results(self, strategy_name: str, result: Dict[str, Any]):
        """
        Emit incremental benchmark results to the client.

        Args:
            strategy_name: Name of the strategy
            result: Benchmark result dictionary
        """
        # Extract relevant metrics
        data = self._prepare_incremental_data(strategy_name, result)

        # Emit to client
        self.socketio.emit("incremental_benchmark_results", data)

        # Add a small delay to ensure messages are processed in order
        time.sleep(0.1)

    def _prepare_incremental_data(self, strategy_name, result):
        """Prepare data for incremental results emission."""
        data = {
            "strategy": strategy_name,
            "makespan": result["makespan"],
            "workload": result["workload"],
            "simulation_time": result.get("simulation_time", 0)
        }

        # Add time-based metrics if available
        incremental_fields = ["time_metrics", "events", "hourly_distribution"]
        for field in incremental_fields:
            if field in result:
                data[field] = result[field]

        return data

    def _emit_results(self, benchmark_results: Dict[str, Any]):
        """
        Emit benchmark results to the client.

        Args:
            benchmark_results: Dictionary of results by strategy
        """
        for strategy_name, result in benchmark_results.items():
            self.socketio.emit("benchmark_results", {
                "strategy": strategy_name,
                "times": result["times"],
                "workload": result["workload"]
            })
            # Add a small delay to ensure messages are processed in order
            time.sleep(0.1)

    def _finalize_benchmark(self):
        """Complete the benchmark and notify the client."""
        self._update_progress(100, "Benchmark complete")
        time.sleep(0.5)  # Give a moment for final progress update
        self.socketio.emit("benchmark_complete", {"success": True})

    def _handle_benchmark_error(self, error: Exception):
        """
        Handle an error during benchmark execution.

        Args:
            error: The exception that occurred
        """
        self.logger.error(f"Error in benchmark: {str(error)}")
        traceback.print_exc()
        self.socketio.emit("benchmark_complete", {"error": str(error)})

    def _update_progress(self, progress: int, current_task: str) -> None:
        """
        Update the benchmark progress and emit a progress event.

        Args:
            progress: Progress percentage (0-100)
            current_task: Description of current task
        """
        self.progress = progress
        elapsed = time.time() - self.start_time

        # Calculate estimated completion
        completion_time = self._calculate_completion_time(progress, elapsed)

        # Emit progress event
        self.socketio.emit("benchmark_progress", {
            "progress": progress,
            "current_task": current_task,
            "elapsed_time": elapsed,
            "estimated_completion": completion_time
        })

    def _calculate_completion_time(self, progress: int, elapsed: float) -> float:
        """
        Calculate estimated completion time.

        Args:
            progress: Current progress (0-100)
            elapsed: Elapsed time in seconds

        Returns:
            float: Estimated remaining time in seconds
        """
        if progress <= 0:
            return 0

        estimated_total = elapsed * 100 / progress
        return estimated_total - elapsed

    def get_available_scenarios(self) -> List[str]:
        """
        Get the list of available benchmark scenarios.

        Returns:
            List: List of scenario names
        """
        return self.model.get_available_scenarios()

    def get_available_strategies(self) -> List[Dict[str, str]]:
        """
        Get information about all available strategies.

        Returns:
            List[Dict[str, str]]: List of strategy information dictionaries
        """
        return self.model.get_available_strategies()

    def add_custom_scenario(self, name: str, requests: List[tuple]) -> List[str]:
        """
        Add a custom scenario for benchmarking.

        Args:
            name: Scenario name
            requests: List of request tuples (origin, destination, urgent)

        Returns:
            List: Updated list of scenario names
        """
        self.model.add_scenario(name, requests)
        return self.get_available_scenarios()

    def get_hourly_rate_data(self):
        """
        Get hourly rate data for time-based benchmarks.
        Returns:
            Dict: Hourly rate data for visualization
        """
        return self.model.get_hourly_rate_data()

    def get_available_time_ranges(self):
        """
        Get available time ranges for time-based benchmarks.

        Returns:
            Dict: Time ranges and hourly rate data
        """
        try:
            # Create a temporary data repository to get the data
            from new_backend_benchmark.execution.repository.transport_data_repository import TransportDataRepository
            repository = TransportDataRepository()

            return {
                "time_ranges": repository.get_available_time_ranges(),
                "hourly_rates": repository.get_hourly_rates_for_chart()
            }
        except Exception as e:
            self.logger.error(f"Error getting time ranges: {str(e)}")
            return {
                "time_ranges": [],
                "hourly_rates": {"labels": [], "data": []}
            }

    def generate_time_scenario(self, start_hour, end_hour, name=None, request_count=None) -> Dict[str, Any]:
        """
        Generate a time-based benchmark scenario.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            name: Name for the scenario
            request_count: Number of requests

        Returns:
            Dict: Result with scenario or error
        """
        try:
            # Validate inputs
            valid, error = self.validate_time_range(start_hour, end_hour)
            if not valid:
                return {"success": False, "error": error}

            # Create a data repository instance for generating requests
            from new_backend_benchmark.execution.repository.transport_data_repository import TransportDataRepository
            repository = TransportDataRepository()

            # Generate a default name if not provided
            if not name:
                # Morning: 5-12, Afternoon: 12-17, Evening: 17-21, Night: 21-5
                if 5 <= int(start_hour) < 12:
                    period = "Morning"
                elif 12 <= int(start_hour) < 17:
                    period = "Afternoon"
                elif 17 <= int(start_hour) < 21:
                    period = "Evening"
                else:
                    period = "Night"
                name = f"{period} {start_hour:02d}-{end_hour:02d}"

            # If request count is provided, use that
            if request_count is not None:
                count = int(request_count)
                generated_requests = repository.generate_benchmark_requests(
                    int(start_hour), int(end_hour), count
                )
            else:
                # Otherwise, get rate and calculate a daily amount
                hourly_rate = repository.get_request_rate(int(start_hour), int(end_hour))
                hours = int(end_hour) - int(start_hour) if int(end_hour) > int(start_hour) else (24 - int(
                    start_hour)) + int(end_hour)

                # Apply daily scaling - divide by approximate days in dataset
                # This transforms the yearly total into a daily average
                estimated_days_in_dataset = 365  # Adjust based on your dataset
                daily_rate = hourly_rate / estimated_days_in_dataset

                # Calculate reasonable number of requests for one day
                count = int(daily_rate * hours)
                count = max(1, min(count, 200))  # Reasonable bounds

                self.logger.info(
                    f"Calculated {count} requests for time range {start_hour}-{end_hour} (rate: {daily_rate:.2f}/hour)")
                generated_requests = repository.generate_benchmark_requests(
                    int(start_hour), int(end_hour), count
                )

            if not generated_requests:
                return {
                    "success": False,
                    "error": f"Failed to generate requests for time range {start_hour}-{end_hour}"
                }

            # Convert to scenario format for the benchmark model
            # The model expects tuples of (origin, destination, urgent)
            scenario_requests = [
                (origin, dest, urgent)
                for origin, dest, _, urgent in generated_requests
            ]

            # Count urgent requests
            urgent_count = sum(1 for _, _, urgent in scenario_requests if urgent)

            # Add to benchmark model
            self.model.add_scenario(name, scenario_requests)

            # Get hourly rate information
            hourly_rate = repository.get_request_rate(int(start_hour), int(end_hour))
            hours = int(end_hour) - int(start_hour) if int(end_hour) > int(start_hour) else (24 - int(
                start_hour)) + int(end_hour)
            daily_rate = hourly_rate / 365  # Convert to daily rate

            # Create complete scenario info
            scenario = {
                "name": name,
                "time_range": f"{int(start_hour):02d}-{int(end_hour):02d}",
                "requests": [
                    {"origin": origin, "destination": dest, "urgent": urgent}
                    for origin, dest, urgent in scenario_requests
                ],
                "urgent_count": urgent_count,
                "request_count": len(scenario_requests),
                "hourly_rate": hourly_rate,
                "daily_rate": daily_rate,
                "hours": hours
            }

            self.logger.info(f"Generated time-based scenario '{name}' with {len(scenario_requests)} requests")
            self.logger.info(
                f"Time range: {start_hour}-{end_hour}, Hourly rate: {hourly_rate:.2f}, Daily rate: {daily_rate:.2f}")

            return {"success": True, "scenario": scenario}

        except Exception as e:
            self.logger.error(f"Error generating time scenario: {str(e)}")
            return {"success": False, "error": str(e)}

    def validate_time_range(self, start_hour, end_hour) -> Tuple[bool, Optional[str]]:
        """
        Validate a time range input.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
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

        # Ensure start and end are different (at least a 1-hour range)
        if start_hour == end_hour:
            return False, "Start and end hour must be different"

        # Valid input
        return True, None

