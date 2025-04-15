

"""
BenchmarkModel for hospital transport system.
Handles standard and incremental benchmark execution.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from new_backend_benchmark.model.time_based_request_generator import TimeBasedRequestGenerator
from Model.simulator_clock import SimulationClock
from new_backend_benchmark.model.strategy_factory import StrategyFactory
from new_backend_benchmark.execution.repository.transport_data_repository import TransportDataRepository


class BenchmarkModel:
    """
    Core model for benchmark operations with support for incremental execution.

    This class handles both standard benchmarks (all requests at once) and
    incremental benchmarks (requests appear over time).
    """

    def __init__(self, hospital_system):
        """Initialize the benchmark model."""
        self.system = hospital_system
        self.scenarios = self._initialize_scenarios()
        self.logger = logging.getLogger("BenchmarkModel")

    def _initialize_scenarios(self) -> Dict[str, List[tuple]]:
        """Initialize default benchmark scenarios."""
        return {
            "Default Scenario": self._get_default_scenario(),
            "Emergency Heavy": self._get_emergency_scenario(),
            "Distributed": self._get_distributed_scenario(),
            "Complex": self._get_complex_benchmark()
        }

    def get_scenario(self, name: str) -> List[tuple]:
        """Get a specific scenario by name."""
        return self.scenarios.get(name, self._get_default_scenario())

    def add_scenario(self, name: str, requests: List[tuple]) -> Dict[str, List[tuple]]:
        """Add a new scenario to the available scenarios."""
        self.scenarios[name] = requests
        return self.scenarios

    def get_available_scenarios(self) -> List[str]:
        """Get names of all available scenarios."""
        return list(self.scenarios.keys())

    def get_available_strategies(self) -> List[Dict[str, str]]:
        """Get information about all available optimization strategies."""
        return StrategyFactory.get_strategy_info()

    def run_benchmark(self, strategy_name: str, num_transporters: int,
                      requests: List[tuple], incremental_mode: bool = False,
                      time_range: Optional[Tuple[float, float]] = None,
                      time_distribution: str = "realistic",
                      **kwargs) -> Dict[str, Any]:
        """
        Run a benchmark with any strategy, optionally in incremental mode.

        Args:
            strategy_name: Name of the strategy to use
            num_transporters: Number of transporters to use
            requests: List of request tuples (origin, destination, urgent)
            incremental_mode: If True, release requests over time
            time_range: Optional tuple (start_time, end_time) for incremental mode
            time_distribution: How to distribute requests in time
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dict: Benchmark results
        """
        # Create transporters and requests
        transporters, created_requests = self._prepare_benchmark_data(num_transporters, requests)

        # Get strategy instance
        strategy = StrategyFactory.get_strategy(strategy_name)

        # Get the graph
        graph = self.system.hospital.get_graph()

        # Special case for Random strategy that needs multiple runs
        if strategy_name == "Random" and not incremental_mode:
            return self._run_random_benchmark(strategy, transporters, created_requests,
                                              graph, runs=kwargs.get("runs", 1))

        # Choose between standard or incremental execution
        if not incremental_mode:
            return self._run_standard_benchmark(strategy, transporters, created_requests, graph)
        else:
            return self._run_incremental_benchmark(
                strategy, transporters, created_requests, graph,
                time_range=time_range,
                time_distribution=time_distribution,
                **kwargs
            )

    def _prepare_benchmark_data(self, num_transporters: int,
                                requests: List[tuple]) -> Tuple[List, List]:
        """Prepare transporters and requests for a benchmark."""
        # Reset system state
        self._reset_system_state()

        # Create transporters
        transporters = self._create_transporters(num_transporters)

        # Create transport requests
        created_requests = self._create_requests(requests)

        return transporters, created_requests

    def _reset_system_state(self):
        """Reset the system state for a new benchmark run."""
        from Model.model_transportation_request import TransportationRequest

        # Clear transporters
        self.system.transport_manager.transporters.clear()

        # Clear all request lists
        TransportationRequest.pending_requests.clear()
        TransportationRequest.ongoing_requests.clear()
        TransportationRequest.completed_requests.clear()

    def _create_transporters(self, num_transporters: int) -> List:
        """Create transporters for the benchmark."""
        transporters = []
        for i in range(num_transporters):
            name = f"Benchmark_T{i + 1}"
            self.system.add_transporter(name)
            transporters.append(self.system.transport_manager.get_transporter(name))
        return transporters

    def _create_requests(self, requests: List[tuple]) -> List:
        """Create transport requests from tuples."""
        created_requests = []
        for origin, destination, urgent in requests:
            req = self.system.create_transport_request(origin, destination, "stretcher", urgent)
            created_requests.append(req)
        return created_requests

    def _run_standard_benchmark(self, strategy, transporters, requests, graph) -> Dict[str, Any]:
        """Run a standard benchmark where all requests are processed at once."""
        # Generate plan with the strategy
        plan = strategy.generate_assignment_plan(transporters, requests, graph)

        # Calculate metrics
        makespan, workload = self._calculate_metrics(transporters, plan, graph)

        return {
            "makespan": makespan,
            "workload": workload,
            "plan": plan,
            "incremental": False
        }

    def _run_random_benchmark(self, strategy, transporters, requests,
                              graph, runs=1) -> Dict[str, Any]:
        """Run multiple random benchmarks."""
        times = []
        workload = {}

        # Run once first to get workload
        plan = strategy.generate_assignment_plan(transporters, requests, graph)
        makespan, workload = self._calculate_metrics(transporters, plan, graph)
        times.append(makespan)

        # Run additional times if needed
        for _ in range(runs - 1):
            plan = strategy.generate_assignment_plan(transporters, requests, graph)
            makespan, _ = self._calculate_metrics(transporters, plan, graph)
            times.append(makespan)

        return {
            "times": times,
            "workload": workload,
            "statistics": self._calculate_statistics(times)
        }

    def _run_incremental_benchmark(self, strategy, transporters, requests, graph,
                                   time_range=None, time_distribution="realistic",
                                   **kwargs) -> Dict[str, Any]:
        """Run a benchmark with incremental request appearance over time."""
        # Set up time range (default to 1 hour if not specified)
        time_range = self._get_time_range(time_range)
        start_time, end_time = time_range

        # Create scheduler for requests
        scheduler = TimeBasedRequestGenerator(
            requests=requests,
            start_time=start_time,
            end_time=end_time,
            time_distribution=time_distribution
        )

        # Set up simulation clock (faster than real-time)
        sim_clock = self._create_simulation_clock()

        # Initialize simulation state
        time_metrics, events, transporter_state = self._initialize_simulation(transporters)

        # Run simulation loop
        self.logger.info(f"Starting incremental benchmark with {len(requests)} requests")
        sim_time = self._run_simulation_loop(
            scheduler, sim_clock, strategy, transporters, graph,
            time_metrics, events, transporter_state
        )

        # Calculate final metrics
        final_metrics = self._calculate_incremental_metrics(transporter_state, sim_time)

        # Get hourly distribution data for visualization
        hourly_distribution = scheduler.get_hourly_distribution_data()

        # Return results
        return {
            "makespan": final_metrics["makespan_estimate"],
            "workload": final_metrics["workload"],
            "incremental": True,
            "time_metrics": time_metrics,
            "events": events,
            "hourly_distribution": hourly_distribution,
            "final_state": transporter_state,
            "simulation_time": sim_time - start_time
        }

    def _get_time_range(self, time_range):
        """Get time range with default if none provided."""
        if not time_range:
            return (0, 3600)  # Default to 1 hour
        return time_range

    def _create_simulation_clock(self):
        """Create and start a simulation clock."""
        sim_clock = SimulationClock(speed_factor=1000)  # 1000x speed
        sim_clock.start()
        return sim_clock

    def _initialize_simulation(self, transporters):
        """Initialize simulation state tracking."""
        # Track metrics over time
        time_metrics = []

        # Event timeline for visualization
        events = []

        # Transporter state tracking
        transporter_state = {
            t.name: {
                "current_request": None,
                "queue": [],
                "completion_time": 0,  # When current task will complete
                "location": t.current_location,
                "total_work": 0
            } for t in transporters
        }

        return time_metrics, events, transporter_state

    def _run_simulation_loop(self, scheduler, sim_clock, strategy, transporters,
                             graph, time_metrics, events, transporter_state):
        """Run the main simulation loop."""
        # Track the start time of the benchmark
        bench_start_time = time.time()
        sim_time = scheduler.start_time

        while not scheduler.all_processed():
            # Get current simulation time
            sim_time = sim_clock.get_time()

            # Bail out if simulation takes too long in real-time
            if self._check_timeout(bench_start_time):
                break

            # Update transporter states
            self._update_transporter_states(transporter_state, sim_time)

            # Process new requests
            self._process_new_requests(
                scheduler, sim_time, strategy, transporters, graph,
                transporter_state, time_metrics, events
            )

            # Advance time to next event
            self._advance_to_next_event(scheduler, transporter_state, sim_time, sim_clock)

            # Small sleep to prevent CPU hogging
            time.sleep(0.001)

        # Stop the clock
        sim_clock.stop()
        self.logger.info(f"Completed incremental benchmark at time {sim_time:.1f}")

        return sim_time

    def _check_timeout(self, bench_start_time, max_duration=300):
        """Check if the benchmark has exceeded the maximum duration."""
        elapsed_real_time = time.time() - bench_start_time
        if elapsed_real_time > max_duration:  # 5 minute max
            self.logger.warning("Benchmark timeout - terminating early")
            return True
        return False

    def _process_new_requests(self, scheduler, sim_time, strategy, transporters,
                              graph, transporter_state, time_metrics, events):
        """Process newly active requests at the current time."""
        # Check for newly active requests
        new_requests = scheduler.get_active_requests(sim_time)

        if not new_requests:
            return

        # Log event
        self.logger.info(f"Processing {len(new_requests)} new requests at time {sim_time:.1f}")

        # Get available transporters
        available_transporters = self._get_available_transporters(transporters, transporter_state)

        # Generate assignment plan for new requests
        target_transporters = available_transporters if available_transporters else transporters
        new_plan = strategy.generate_assignment_plan(target_transporters, new_requests, graph)

        # Apply new assignments
        self._apply_incremental_assignments(
            transporter_state, new_plan, new_requests, graph, sim_time
        )

        # Record event
        events.append({
            "time": sim_time,
            "type": "new_requests",
            "count": len(new_requests)
        })

        # Calculate current metrics
        current_metrics = self._calculate_incremental_metrics(
            transporter_state, sim_time
        )

        # Add to time metrics
        time_metrics.append({
            "time": sim_time,
            "active_requests": len(new_requests),
            "available_transporters": len(available_transporters),
            "makespan_estimate": current_metrics["makespan_estimate"],
            "workload": current_metrics["workload"]
        })

    def _get_available_transporters(self, transporters, transporter_state):
        """Get transporters that are available for new assignments."""
        return [
            t for t in transporters
            if transporter_state[t.name]["current_request"] is None
        ]

    def _advance_to_next_event(self, scheduler, transporter_state, sim_time, sim_clock):
        """Advance simulation time to the next event."""
        next_time = scheduler.get_next_request_time()
        if next_time is None:
            return

        # Find next event time (either next request or next transporter completion)
        completion_times = [
            state["completion_time"] for state in transporter_state.values()
            if state["current_request"] is not None
        ]

        if completion_times:
            next_event_time = min([next_time] + [t for t in completion_times if t > sim_time])
        else:
            next_event_time = next_time

        # Advance simulation clock
        time_diff = next_event_time - sim_time
        if time_diff > 0:
            sim_clock.advance_time(seconds=time_diff)

    def _update_transporter_states(self, transporter_state, current_time):
        """Update transporter states based on current time."""
        for t_name, state in transporter_state.items():
            # Skip if no active request
            if state["current_request"] is None:
                continue

            # Check if current request is complete
            if current_time >= state["completion_time"]:
                self._complete_current_request(transporter_state, t_name, current_time)

    def _complete_current_request(self, transporter_state, t_name, current_time):
        """Complete the current request and start the next one if queued."""
        state = transporter_state[t_name]

        # Current request is complete
        completed_request = state["current_request"]

        # Update transporter location to request destination
        state["location"] = completed_request.destination

        # Mark request as complete
        state["current_request"] = None

        # If there are queued requests, start the next one
        if state["queue"]:
            next_request = state["queue"].pop(0)
            self._assign_request_to_transporter(
                transporter_state, t_name, next_request, current_time
            )

    def _apply_incremental_assignments(self, transporter_state, plan, requests, graph, current_time):
        """Apply new assignments to transporters."""
        # Apply assignments from the plan
        for t_name, assigned_requests in plan.items():
            for request in assigned_requests:
                if request in requests:  # Only process new requests
                    if transporter_state[t_name]["current_request"] is None:
                        # Transporter is available, assign directly
                        self._assign_request_to_transporter(
                            transporter_state, t_name, request, current_time
                        )
                    else:
                        # Transporter is busy, add to queue
                        transporter_state[t_name]["queue"].append(request)

    def _assign_request_to_transporter(self, transporter_state, t_name, request, current_time):
        """Assign a request to a transporter and calculate completion time."""
        state = transporter_state[t_name]

        # Calculate time to reach request origin from current location
        origin_time = self._calculate_travel_time_between(
            state["location"], request.origin
        )

        # Calculate time to transport from origin to destination
        destination_time = self._calculate_travel_time_between(
            request.origin, request.destination
        )

        # Calculate total time for this request
        total_time = origin_time + destination_time

        # Update state
        state["current_request"] = request
        state["completion_time"] = current_time + total_time
        state["total_work"] += total_time

    def _calculate_travel_time_between(self, start, end):
        """Calculate travel time between two locations."""
        try:
            # Use the actual hospital pathfinder
            path, distance = self.system.hospital.pathfinder.dijkstra(start, end)
            return distance
        except (AttributeError, ValueError) as e:
            self.logger.warning(f"Error calculating travel time: {str(e)}")
            # Fallback to simple distance estimate
            return 5.0 + (hash(start) % 5) + (hash(end) % 5)

    def _calculate_incremental_metrics(self, transporter_state, current_time):
        """Calculate current metrics from transporter state."""
        # Calculate makespan estimate
        completion_times = self._get_completion_times(transporter_state)

        # Calculate makespan as the latest completion time
        makespan_estimate = max(completion_times) if completion_times else current_time

        # Calculate current workload for each transporter
        workload = {
            t_name: state["total_work"]
            for t_name, state in transporter_state.items()
        }

        return {
            "makespan_estimate": makespan_estimate,
            "workload": workload
        }

    def _get_completion_times(self, transporter_state):
        """Calculate expected completion times for all transporters."""
        completion_times = []

        for t_name, state in transporter_state.items():
            # If transporter has a current request, use its completion time
            if state["current_request"] is not None:
                completion_times.append(state["completion_time"])

                # Also account for queued requests
                if state["queue"]:
                    queue_completion_time = self._estimate_queue_completion_time(state)
                    completion_times.append(queue_completion_time)

        return completion_times

    def _estimate_queue_completion_time(self, state):
        """Estimate completion time for all queued requests."""
        completion_time = state["completion_time"]
        current_location = state["current_request"].destination

        for queued_request in state["queue"]:
            # Estimate travel from previous destination to next origin
            origin_time = self._calculate_travel_time_between(
                current_location, queued_request.origin
            )

            # Estimate travel from origin to destination
            destination_time = self._calculate_travel_time_between(
                queued_request.origin, queued_request.destination
            )

            # Update completion time and current location
            completion_time += origin_time + destination_time
            current_location = queued_request.destination

        return completion_time

    def _calculate_metrics(self, transporters: List, plan: Dict,
                           graph) -> Tuple[float, Dict[str, float]]:
        """Calculate makespan and workload metrics from a plan."""
        makespan = 0
        workload = {}

        for t in transporters:
            assigned_requests = plan.get(t.name, [])
            total_time = self._estimate_execution_time(t, assigned_requests, graph)
            workload[t.name] = total_time
            makespan = max(makespan, total_time)

        return makespan, workload

    def _estimate_execution_time(self, transporter, requests, graph) -> float:
        """Estimate the time a transporter would take to complete all assigned requests."""
        time = 0
        current_location = transporter.current_location

        for request in requests:
            # Travel to request origin
            time += self._calculate_travel_time(current_location, request.origin,
                                                transporter.pathfinder, graph)

            # Travel to request destination
            time += self._calculate_travel_time(request.origin, request.destination,
                                                transporter.pathfinder, graph)

            current_location = request.destination

        return time

    def _calculate_travel_time(self, start, end, pathfinder, graph) -> float:
        """Calculate travel time between two points using pathfinder."""
        path, _ = pathfinder.dijkstra(start, end)
        return sum(
            graph.get_edge_weight(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    def _calculate_statistics(self, times: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of completion times."""
        if not times:
            return {
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0
            }

        return {
            "mean": float(np.mean(times)),
            "median": float(np.median(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times))
        }

    def calculate_workload_statistics(self, workload: Dict[str, float]) -> Dict[str, float]:
        """Calculate workload distribution statistics."""
        if not workload:
            return {
                "std": 0,
                "mean": 0,
                "min": 0,
                "max": 0
            }

        values = list(workload.values())
        return {
            "std": float(np.std(values)),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

    def _get_default_scenario(self) -> List[tuple]:
        """Default transport scenario with a mix of requests."""
        return [
            ("Emergency", "ICU", True),
            ("Reception", "Radiology", False),
            ("ICU", "General Ward", False),
            ("Cardiology", "Surgery", False),
            ("Pharmacy", "Neurology", False),
            ("Pediatrics", "Orthopedics", True),
            ("Admin Office", "Cafeteria", False),
            ("Radiology", "Laboratory", False),
            ("Emergency", "Surgery", True),
            ("Reception", "Cardiology", False)
        ]

    def _get_emergency_scenario(self) -> List[tuple]:
        """Scenario focused on emergency department requests."""
        return [
            ("Emergency", "ICU", True),
            ("Emergency", "Surgery", True),
            ("Emergency", "Radiology", True),
            ("Emergency", "General Ward", False),
            ("Emergency", "Pharmacy", False),
            ("ICU", "Emergency", False)
        ]

    def _get_distributed_scenario(self) -> List[tuple]:
        """Scenario with evenly distributed requests across departments."""
        return [
            ("Reception", "Radiology", False),
            ("Radiology", "Laboratory", False),
            ("Laboratory", "Pharmacy", False),
            ("Pharmacy", "ICU", False),
            ("ICU", "Emergency", True),
            ("Emergency", "Surgery", True),
            ("Surgery", "Cardiology", False),
            ("Cardiology", "Neurology", False),
            ("Neurology", "Orthopedics", False),
            ("Orthopedics", "Pediatrics", False),
            ("Pediatrics", "General Ward", False),
            ("General Ward", "ICU", True),
            ("Admin Office", "Reception", False),
            ("Cafeteria", "Admin Office", False),
            ("Emergency", "Cafeteria", False)
        ]

    def _get_complex_benchmark(self) -> List[tuple]:
        """Complex benchmark scenario with 25 transport requests."""
        return [
            # Emergency department requests (some urgent)
            ("Emergency", "ICU", True),
            ("Emergency", "Surgery", True),
            ("Emergency", "Radiology", True),
            ("Emergency", "General Ward", False),
            ("Emergency", "Pharmacy", False),

            # ICU requests
            ("ICU", "Surgery", True),
            ("ICU", "Radiology", False),
            ("ICU", "Pharmacy", False),

            # Surgery department requests
            ("Surgery", "ICU", True),
            ("Surgery", "Recovery", False),
            ("Surgery", "General Ward", False),

            # Radiology requests
            ("Radiology", "Emergency", True),
            ("Radiology", "Oncology", False),
            ("Radiology", "Neurology", False),

            # Other specialized departments
            ("Cardiology", "ICU", True),
            ("Neurology", "Surgery", False),
            ("Orthopedics", "Radiology", False),
            ("Pediatrics", "Emergency", True),
            ("Oncology", "Radiology", False),

            # Support services
            ("Laboratory", "Emergency", True),
            ("Pharmacy", "General Ward", False),
            ("General Ward", "Radiology", False),
            ("Reception", "Cardiology", False),
            ("Cafeteria", "Admin Office", False),
            ("Admin Office", "Reception", False)
        ]

    # In BenchmarkModel
    def get_hourly_rate_data(self) -> Dict[str, Any]:
        """
        Get hourly rate data for time-based benchmarks.

        Returns:
            Dict: Hourly rate data for visualization
        """
        try:
            repository = TransportDataRepository()

            hourly_rates = repository.get_hourly_rates_for_chart()

            result = {
                "hourly_rates": hourly_rates,
                "available_time_ranges": repository.get_available_time_ranges(),
                "metadata": {
                    "total_yearly_requests": sum(hourly_rates.get("data", [])),
                    "daily_average": sum(hourly_rates.get("data", [])) / 365
                }
            }

            return result
        except Exception as e:
            logging.error(f"Error getting hourly rate data: {str(e)}")
            return {
                "hourly_rates": {"labels": [], "data": []},
                "available_time_ranges": [],
                "metadata": {"error": str(e)}
            }