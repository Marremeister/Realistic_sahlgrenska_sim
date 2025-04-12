"""
Model component for benchmark functionality.
Handles core benchmark data and operations.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from Model.model_transportation_request import TransportationRequest
from .strategy_factory import StrategyFactory


class BenchmarkModel:
    """
    Core model for benchmark operations.
    Manages benchmark scenarios and coordinates optimization strategies.
    """

    def __init__(self, hospital_system):
        """
        Initialize the benchmark model with a reference to the hospital system.

        Args:
            hospital_system: The hospital system to benchmark
        """
        self.system = hospital_system
        self.scenarios = self._initialize_scenarios()

    def _initialize_scenarios(self) -> Dict[str, List[tuple]]:
        """
        Initialize default benchmark scenarios.

        Returns:
            dict: A dictionary of predefined scenarios
        """
        return {
            "Default Scenario": self._get_default_scenario(),
            "Emergency Heavy": self._get_emergency_scenario(),
            "Distributed": self._get_distributed_scenario(),
            "Complex": self._get_complex_benchmark()
        }

    def get_scenario(self, name: str) -> List[tuple]:
        """
        Get a specific scenario by name.

        Args:
            name: Name of the scenario

        Returns:
            List[tuple]: List of (origin, destination, urgent) request tuples
        """
        return self.scenarios.get(name, self._get_default_scenario())

    def add_scenario(self, name: str, requests: List[tuple]) -> Dict[str, List[tuple]]:
        """
        Add a new scenario to the available scenarios.

        Args:
            name: Scenario name
            requests: List of request tuples (origin, destination, urgent)

        Returns:
            Dict: Updated dictionary of available scenarios
        """
        self.scenarios[name] = requests
        return self.scenarios

    def get_available_scenarios(self) -> List[str]:
        """
        Get names of all available scenarios.

        Returns:
            List[str]: List of scenario names
        """
        return list(self.scenarios.keys())

    def get_available_strategies(self) -> List[Dict[str, str]]:
        """
        Get information about all available optimization strategies.

        Returns:
            List[Dict[str, str]]: List with information about each strategy
        """
        return StrategyFactory.get_strategy_info()

    def run_benchmark(self, strategy_name: str, num_transporters: int,
                     requests: List[tuple], **kwargs) -> Dict[str, Any]:
        """
        Run a benchmark with any strategy.

        Args:
            strategy_name: Name of the strategy to use
            num_transporters: Number of transporters to use
            requests: List of request tuples (origin, destination, urgent)
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

        # Handle special case for Random strategy that needs multiple runs
        if strategy_name == "Random":
            return self._run_random_benchmark(strategy, transporters, created_requests,
                                             graph, runs=kwargs.get("runs", 1))

        # Generate plan with the strategy
        plan = strategy.generate_assignment_plan(transporters, created_requests, graph)

        # Calculate metrics
        makespan, workload = self._calculate_metrics(transporters, plan, graph)

        return {
            "makespan": makespan,
            "workload": workload,
            "plan": plan
        }

    def _prepare_benchmark_data(self, num_transporters: int,
                               requests: List[tuple]) -> Tuple[List, List]:
        """
        Prepare transporters and requests for a benchmark.

        Args:
            num_transporters: Number of transporters to use
            requests: List of request tuples (origin, destination, urgent)

        Returns:
            Tuple[List, List]: Tuple of (transporters, created_requests)
        """
        # Reset system state
        self._reset_system_state()

        # Create transporters
        transporters = []
        for i in range(num_transporters):
            name = f"Benchmark_T{i + 1}"
            self.system.add_transporter(name)
            transporters.append(self.system.transport_manager.get_transporter(name))

        # Create transport requests
        created_requests = []
        for origin, destination, urgent in requests:
            req = self.system.create_transport_request(origin, destination, "stretcher", urgent)
            created_requests.append(req)

        return transporters, created_requests

    def _run_random_benchmark(self, strategy, transporters, requests,
                             graph, runs=1) -> Dict[str, Any]:
        """
        Run multiple random benchmarks.

        Args:
            strategy: Random strategy instance
            transporters: List of transporters
            requests: List of requests
            graph: Hospital graph
            runs: Number of runs to perform

        Returns:
            Dict: Results including times and workload
        """
        times = []
        workload = {}

        # Run once first to get workload (optimization)
        plan = strategy.generate_assignment_plan(transporters, requests, graph)
        makespan, workload = self._calculate_metrics(transporters, plan, graph)
        times.append(makespan)

        # Run additional times (if needed)
        for _ in range(runs - 1):
            # Generate a new random plan
            plan = strategy.generate_assignment_plan(transporters, requests, graph)
            # Calculate makespan only (skip workload for performance)
            makespan, _ = self._calculate_metrics(transporters, plan, graph)
            times.append(makespan)

        return {
            "times": times,
            "workload": workload,
            "statistics": self._calculate_statistics(times)
        }

    def _calculate_metrics(self, transporters: List, plan: Dict,
                          graph) -> Tuple[float, Dict[str, float]]:
        """
        Calculate makespan and workload metrics from a plan.

        Args:
            transporters: List of transporters
            plan: Assignment plan
            graph: Hospital graph

        Returns:
            Tuple[float, Dict[str, float]]: Makespan and workload dictionary
        """
        makespan = 0
        workload = {}

        for t in transporters:
            assigned_requests = plan.get(t.name, [])
            total_time = self._estimate_execution_time(t, assigned_requests, graph)
            workload[t.name] = total_time
            makespan = max(makespan, total_time)

        return makespan, workload

    def _reset_system_state(self):
        """Reset the system state for a new benchmark run."""
        # Clear transporters
        self.system.transport_manager.transporters.clear()

        # Clear all request lists
        TransportationRequest.pending_requests.clear()
        TransportationRequest.ongoing_requests.clear()
        TransportationRequest.completed_requests.clear()

    def _estimate_execution_time(self, transporter, requests, graph) -> float:
        """
        Estimate the time a transporter would take to complete all assigned requests.

        Args:
            transporter: The transporter object
            requests: List of request objects
            graph: Hospital graph

        Returns:
            float: Estimated completion time in seconds
        """
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
        """
        Calculate travel time between two points.

        Args:
            start: Start location
            end: End location
            pathfinder: Pathfinder to use
            graph: Hospital graph

        Returns:
            float: Travel time
        """
        path, _ = pathfinder.dijkstra(start, end)
        return sum(
            graph.get_edge_weight(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    def _calculate_statistics(self, times: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of completion times.

        Args:
            times: List of completion times

        Returns:
            Dict: Dictionary of statistics
        """
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
        """
        Calculate workload distribution statistics.

        Args:
            workload: Dictionary mapping transporter names to times

        Returns:
            Dict: Dictionary of statistics
        """
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

    # Legacy method for compatibility with existing code
    def run_ilp_benchmark(self, num_transporters: int, requests: List[tuple],
                        ilp_mode, extra_params=None) -> Dict[str, Any]:
        """
        Run a benchmark using an ILP strategy.

        This is a compatibility method for existing code.

        Args:
            num_transporters: Number of transporters to use
            requests: List of transport requests
            ilp_mode: ILP optimization mode
            extra_params: Additional parameters for the ILP optimizer

        Returns:
            Dict: Benchmark results
        """
        from Model.Assignment_strategies.ILP.ilp_mode import ILPMode

        # Map ILP mode to strategy name
        strategy_map = {
            ILPMode.MAKESPAN: "ILP: Makespan",
            ILPMode.EQUAL_WORKLOAD: "ILP: Equal Workload",
            ILPMode.URGENCY_FIRST: "ILP: Urgency First",
            ILPMode.CLUSTER_BASED: "ILP: Cluster-Based"
        }

        strategy_name = strategy_map.get(ilp_mode, "ILP: Makespan")
        return self.run_benchmark(strategy_name, num_transporters, requests, **(extra_params or {}))

    # Legacy method for compatibility with existing code
    def run_genetic_benchmark(self, num_transporters: int, requests: List[tuple],
                            params=None) -> Dict[str, Any]:
        """
        Run a benchmark using genetic algorithm optimization.

        This is a compatibility method for existing code.

        Args:
            num_transporters: Number of transporters to use
            requests: List of transport requests
            params: Parameters for the genetic algorithm

        Returns:
            Dict: Benchmark results
        """
        return self.run_benchmark("Genetic Algorithm", num_transporters, requests, **(params or {}))

    # Legacy method for compatibility with existing code
    def run_random_benchmark(self, num_transporters: int, requests: List[tuple],
                          runs=1) -> List[Dict[str, Any]]:
        """
        Run random assignment benchmarks.

        This is a compatibility method for existing code.

        Args:
            num_transporters: Number of transporters to use
            requests: List of transport requests
            runs: Number of runs to perform

        Returns:
            List: List of result dictionaries
        """
        result = self.run_benchmark("Random", num_transporters, requests, runs=runs)

        # Convert to the expected format
        return [
            {"makespan": result["times"][i], "workload": result["workload"] if i == 0 else None}
            for i in range(len(result["times"]))
        ]

    # Predefined scenario methods

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