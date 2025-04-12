"""
Controller component for benchmark functionality.
Coordinates between the model and view, handling user interactions.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from new_backend_benchmark.strategies.time_based_benchmark_model import TimeBasedBenchmarkModel


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

        # Initialize time model (only when needed to avoid circular imports)
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

    def _ensure_time_model(self):
        """
        Ensure the time-based model is loaded.
        Lazy loading to avoid circular imports.
        """
        if self.time_model is None:

            self.time_model = TimeBasedBenchmarkModel(self.model)

    def start_benchmark(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Start a benchmark with the given configuration.

        Args:
            config: Benchmark configuration including:
                - transporters: Number of transporters to use
                - random_runs: Number of random simulations to run
                - strategies: List of strategies to benchmark
                - scenarios: List of scenarios to use

        Returns:
            Dict: Status message
        """
        try:
            # Extract configuration
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
        validated = {
            "transporters": config.get("transporters", 3),
            "random_runs": config.get("random_runs", 100),
            "strategies": config.get("strategies", ["ILP: Makespan", "Random"]),
            "scenarios": config.get("scenarios", ["Default Scenario"])
        }

        # Validate transporters (must be > 0)
        if validated["transporters"] <= 0:
            raise ValueError("Number of transporters must be greater than 0")

        # Validate random runs (must be > 0)
        if validated["random_runs"] <= 0:
            validated["random_runs"] = 100

        # Validate strategies (must be in available strategies)
        available_strategies = [s["name"] for s in self.model.get_available_strategies()]
        validated["strategies"] = [s for s in validated["strategies"] if s in available_strategies]
        if not validated["strategies"]:
            validated["strategies"] = ["ILP: Makespan", "Random"]

        # Validate scenarios (must be in available scenarios)
        available_scenarios = self.model.get_available_scenarios()
        validated["scenarios"] = [s for s in validated["scenarios"] if s in available_scenarios]
        if not validated["scenarios"]:
            validated["scenarios"] = ["Default Scenario"]

        return validated

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
                config["scenarios"]
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

    def _run_benchmark_thread(self, num_transporters: int, random_runs: int,
                              strategy_names: List[str], scenario_names: List[str]) -> None:
        """
        Run the benchmark in a background thread.

        Args:
            num_transporters: Number of transporters to use
            random_runs: Number of random simulations to run
            strategy_names: List of strategy names to benchmark
            scenario_names: List of scenario names to use
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
                    scenario_name
                )

                # Emit results
                self._emit_results(benchmark_results)

                # Emit benchmark complete
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

    def _run_all_strategies(self, results: Dict[str, Any], strategy_names: List[str],
                            num_transporters: int, requests: List[tuple],
                            random_runs: int, scenario_name: str):
        """
        Run all requested strategies for a scenario.

        Args:
            results: Dictionary to store results
            strategy_names: List of strategy names to run
            num_transporters: Number of transporters to use
            requests: List of requests for the scenario
            random_runs: Number of random runs
            scenario_name: Name of the current scenario
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
            self._update_progress(
                progress_base,
                f"Running {strategy_name} optimization for {scenario_name}"
            )

            # Run the appropriate strategy
            if strategy_name == "Random":
                self._run_random_strategy(
                    results, num_transporters, requests, random_runs,
                    progress_base, i, total_strategies
                )
            else:
                self._run_single_strategy(
                    results, strategy_name, num_transporters, requests
                )

    def _run_random_strategy(self, results: Dict[str, Any], num_transporters: int,
                             requests: List[tuple], random_runs: int,
                             progress_base: int, strategy_index: int, total_strategies: int):
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

    def _update_random_progress(self, random_runs: int, progress_base: int,
                                strategy_index: int, total_strategies: int):
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

    def _run_single_strategy(self, results: Dict[str, Any], strategy_name: str,
                             num_transporters: int, requests: List[tuple]):
        """
        Run a single optimization strategy.

        Args:
            results: Dictionary to store results
            strategy_name: Name of the strategy to run
            num_transporters: Number of transporters to use
            requests: List of requests
        """
        try:
            # Run the benchmark
            result = self.model.run_benchmark(
                strategy_name,
                num_transporters,
                requests
            )

            # Store results
            results[strategy_name] = {
                "times": [result["makespan"]],
                "workload": result["workload"]
            }

        except Exception as e:
            self.logger.error(f"Error in {strategy_name} benchmark: {str(e)}")
            results[strategy_name] = {
                "times": [0],
                "workload": {},
                "error": str(e)
            }

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
        import traceback
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

    # Time-based benchmark methods

    def get_available_time_ranges(self) -> Dict[str, Any]:
        """
        Get available time ranges and hourly rates for time-based benchmarking.

        Returns:
            Dict: Dictionary with time ranges and hourly rates
        """
        self._ensure_time_model()
        return {
            "time_ranges": self.time_model.get_available_time_ranges(),
            "hourly_rates": self.time_model.get_hourly_rates_data()
        }

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

        # Valid input
        return True, None

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
        # Validate inputs
        valid, error = self.validate_time_range(start_hour, end_hour)
        if not valid:
            return {"success": False, "error": error}

        # Generate scenario
        self._ensure_time_model()
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

    def run_time_based_benchmark(self, start_hour, end_hour, transporters, random_runs=100) -> Dict[str, Any]:
        """
        Run a time-based benchmark.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            transporters: Number of transporters
            random_runs: Number of random runs for comparison

        Returns:
            Dict: Benchmark results
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
        self._ensure_time_model()
        result = self.time_model.run_benchmark_for_time_range(
            int(start_hour), int(end_hour), transporters, random_runs
        )

        if "error" in result:
            return {"success": False, "error": result["error"]}

        return {"success": True, "data": result}