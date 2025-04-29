from abc import ABC, abstractmethod
import pulp
import time
import numpy as np
from copy import deepcopy


class ILPCore(ABC):
    def __init__(self, transporters, requests, graph, use_two_phase=True,
                 time_limit=30, use_warm_start=True):
        """
        Initialize the enhanced ILP core.

        Args:
            transporters: List of transporter objects
            requests: List of transport request objects
            graph: Hospital graph object
            use_two_phase: Whether to use the two-phase approach (assignment then sequencing)
            time_limit: Maximum time in seconds for optimization
            use_warm_start: Whether to use warm starting for the solver
        """
        self.transporters = transporters
        self.requests = requests
        self.graph = graph
        self.use_two_phase = use_two_phase
        self.time_limit = time_limit
        self.use_warm_start = use_warm_start
        self.model = pulp.LpProblem("Transport_Assignment", pulp.LpMinimize)
        self.assign_vars = {}

        # Performance metrics
        self.phase1_time = 0
        self.phase2_time = 0
        self.total_time = 0

        # Problem size threshold for two-phase approach
        self.two_phase_threshold = 10  # Use two-phase if more than 20 requests

    def build_and_solve(self):
        """
        Master method to build and solve the optimization problem.
        Fixed to properly track timing in both standard and two-phase approaches.
        """
        # Store start time as instance variable to ensure it's available throughout
        self.optimization_start_time = time.time()
        print(f"Starting optimization with {len(self.requests)} requests (threshold: {self.two_phase_threshold})")

        try:
            # Initialize timing variables
            self.phase1_time = 0
            self.phase2_time = 0

            # For small problems or if two-phase is disabled, use standard approach
            if len(self.requests) <= self.two_phase_threshold or not self.use_two_phase:
                print("Using standard approach")
                self.define_variables()
                self.add_constraints()
                self.define_objective()

                # Apply warm start if enabled
                if self.use_warm_start:
                    warm_start = self._generate_warm_start()
                    if warm_start:
                        for (t_name, r_id), assignment in warm_start.items():
                            if (t_name, r_id) in self.assign_vars:
                                self.assign_vars[(t_name, r_id)].setInitialValue(assignment)

                # Solve with time limit
                solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit)
                self.model.solve(solver)

                assignment_plan = self.extract_assignments()
            else:
                print("Using two-phase approach")
                # Phase 1: Initial assignment
                phase1_start = time.time()
                print("Starting Phase 1")
                initial_assignments = self._phase1_assignment()
                self.phase1_time = time.time() - phase1_start
                print(f"Phase 1 completed in {self.phase1_time:.2f}s")

                # Phase 2: Sequence optimization per transporter
                phase2_start = time.time()
                print("Starting Phase 2")
                assignment_plan = self._phase2_sequencing(initial_assignments)
                self.phase2_time = time.time() - phase2_start
                print(f"Phase 2 completed in {self.phase2_time:.2f}s")

            # Calculate total time - ensure this happens for both paths
            self.total_time = time.time() - self.optimization_start_time
            print(
                f"Total optimization time: {self.total_time:.2f}s (Phase 1: {self.phase1_time:.2f}s, Phase 2: {self.phase2_time:.2f}s)")

            return assignment_plan

        except Exception as e:
            # Catch any exceptions to ensure we still calculate timing
            import traceback
            print(f"Error during optimization: {str(e)}")
            traceback.print_exc()

            # Still calculate total time even if there was an error
            self.total_time = time.time() - self.optimization_start_time
            print(f"Total time before error: {self.total_time:.2f}s")

            # Return empty plan as fallback
            return {t.name: [] for t in self.transporters}

    def define_variables(self):
        """Define the decision variables for the ILP model."""
        for t in self.transporters:
            for r in self.requests:
                var_name = f"x_{t.name}_{r.id}"
                self.assign_vars[(t.name, r.id)] = pulp.LpVariable(var_name, cat="Binary")

    def add_constraints(self):
        """Add basic constraints to the ILP model."""
        # Each request must be assigned to exactly one transporter
        for r in self.requests:
            self.model += (
                pulp.lpSum(self.assign_vars[(t.name, r.id)] for t in self.transporters) == 1,
                f"UniqueAssignment_{r.id}"
            )

    @abstractmethod
    def define_objective(self):
        """
        Define the objective function for optimization.
        Must be implemented by subclasses to specify their optimization goal.
        """
        pass

    def _phase1_assignment(self):
        """
        Phase 1: Initial assignment of requests to transporters.
        Uses a simplified version of the ILP model.

        Returns:
            dict: Initial assignment plan
        """
        print("Phase 1: Performing initial transporter assignment...")

        # Generate warm start solution if enabled
        warm_start = None
        if self.use_warm_start:
            warm_start = self._generate_warm_start()

        # Create the model for Phase 1
        self.define_variables()
        self.add_constraints()
        self.define_objective()

        # Apply warm start if available
        if warm_start and self.use_warm_start:
            for (t_name, r_id), assignment in warm_start.items():
                if (t_name, r_id) in self.assign_vars:
                    self.assign_vars[(t_name, r_id)].setInitialValue(assignment)

        # Solve with time limit
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit // 2)
        self.model.solve(solver)

        # Extract initial assignments
        initial_plan = {t.name: [] for t in self.transporters}
        for (t_name, r_id), var in self.assign_vars.items():
            if var.value() == 1:
                req = next((r for r in self.requests if r.id == r_id), None)
                if req:
                    initial_plan[t_name].append(req)

        return initial_plan

    def _phase2_sequencing(self, initial_assignments):
        """
        Phase 2: Optimize the sequence of requests for each transporter.
        Added detailed debugging to track where time is spent.

        Args:
            initial_assignments: Dict mapping transporter names to assigned requests

        Returns:
            dict: Final assignment plan with optimized sequences
        """
        print("=== Phase 2: Optimizing request sequences ===")
        final_plan = {t.name: [] for t in self.transporters}

        # Debug info
        total_requests = sum(len(reqs) for reqs in initial_assignments.values())
        print(f"Sequencing {total_requests} requests across {len(initial_assignments)} transporters")

        # Track time for each transporter
        transporter_times = {}

        # Process each transporter's assignments
        for t in self.transporters:
            t_start = time.time()
            print(f"Processing transporter {t.name}")

            t_requests = initial_assignments.get(t.name, [])
            if not t_requests:
                print(f"  No requests for {t.name}, skipping")
                continue

            print(f"  Sequencing {len(t_requests)} requests")

            # Choose optimization method based on request count
            if len(t_requests) <= 10:
                print(f"  Using dynamic programming for {len(t_requests)} requests")
                try:
                    dp_start = time.time()
                    final_plan[t.name] = self._optimize_sequence_dp(t, t_requests)
                    dp_time = time.time() - dp_start
                    print(f"  DP sequencing completed in {dp_time:.4f}s")
                except Exception as e:
                    print(f"  Error in DP sequencing: {str(e)}")
                    # Fallback to greedy approach
                    print(f"  Falling back to greedy chain")
                    final_plan[t.name] = self.sort_requests_by_greedy_chain(t, t_requests)
            else:
                # For larger sets, use ILP for sequencing
                print(f"  Using ILP for {len(t_requests)} requests")
                try:
                    ilp_start = time.time()
                    final_plan[t.name] = self._optimize_sequence_ilp(t, t_requests)
                    ilp_time = time.time() - ilp_start
                    print(f"  ILP sequencing completed in {ilp_time:.4f}s")
                except Exception as e:
                    print(f"  Error in ILP sequencing: {str(e)}")
                    # Fallback to greedy approach
                    print(f"  Falling back to greedy chain")
                    final_plan[t.name] = self.sort_requests_by_greedy_chain(t, t_requests)

            t_time = time.time() - t_start
            transporter_times[t.name] = t_time
            print(f"  Transporter {t.name} processing completed in {t_time:.4f}s")

        # Show summary
        total_phase2 = sum(transporter_times.values())
        print(f"All transporters processed, total processing time: {total_phase2:.4f}s")
        print("=== End Phase 2 ===")

        return final_plan

    def _optimize_sequence_dp(self, transporter, requests):
        """
        Optimize request sequence using dynamic programming approach.
        Efficient for small to medium sized request sets.

        Args:
            transporter: Transporter object
            requests: List of assigned requests

        Returns:
            list: Optimally sequenced requests
        """
        if not requests:
            return []

        n = len(requests)
        if n == 1:
            return requests

        # Calculate pairwise travel times including from current location
        travel_times = {}
        current_loc = transporter.current_location

        # Time from current location to each request origin
        for req in requests:
            travel_times[(current_loc, req.origin)] = self.estimate_point_to_point_time(
                current_loc, req.origin)

        # Time between requests (from destination to next origin)
        for req1 in requests:
            for req2 in requests:
                if req1 != req2:
                    travel_times[(req1.destination, req2.origin)] = self.estimate_point_to_point_time(
                        req1.destination, req2.origin)

        # Time to complete each request (origin to destination)
        for req in requests:
            travel_times[(req.origin, req.destination)] = self.estimate_point_to_point_time(
                req.origin, req.destination)

        # Dynamic programming setup for TSP-like problem
        # State: (visited_requests, last_location)
        # Value: minimum time to visit all requests in visited_requests ending at last_location

        # Initialize DP table
        dp = {}

        # Base case: starting from current location with no requests visited
        dp[(frozenset(), current_loc)] = 0

        # Build up subsets of increasing size
        for size in range(1, n + 1):
            for subset in self._get_subsets(requests, size):
                frozen_subset = frozenset(subset)

                for req in subset:
                    # Consider this request as the last one
                    prev_subset = frozenset(r for r in subset if r != req)

                    # Check all possible previous locations
                    for prev_loc in [r.destination for r in requests if r in prev_subset] + [current_loc]:
                        if (prev_subset, prev_loc) in dp:
                            # Time to go from previous location to this request's origin
                            time_to_origin = travel_times.get((prev_loc, req.origin), float('inf'))

                            # Time to complete this request
                            time_to_complete = travel_times.get((req.origin, req.destination), float('inf'))

                            # Total time for this sequence
                            total_time = dp[(prev_subset, prev_loc)] + time_to_origin + time_to_complete

                            # Update if better
                            if (frozen_subset, req.destination) not in dp or total_time < dp[
                                (frozen_subset, req.destination)]:
                                dp[(frozen_subset, req.destination)] = total_time

        # Reconstruct optimal sequence
        current_subset = frozenset(requests)
        last_loc = None
        min_time = float('inf')

        # Find ending location with minimum time
        for req in requests:
            if (current_subset, req.destination) in dp and dp[(current_subset, req.destination)] < min_time:
                min_time = dp[(current_subset, req.destination)]
                last_loc = req.destination
                last_req = req

        # Reconstruct path
        path = []
        while current_subset:
            path.append(last_req)
            current_subset = frozenset(r for r in current_subset if r != last_req)

            # Find previous request
            prev_req = None
            prev_min_time = float('inf')

            for req in current_subset:
                if (current_subset, req.destination) in dp:
                    time_to_last = travel_times.get((req.destination, last_req.origin), float('inf'))
                    total_time = dp[(current_subset, req.destination)] + time_to_last

                    if total_time < prev_min_time:
                        prev_min_time = total_time
                        prev_req = req

            if prev_req:
                last_req = prev_req
                last_loc = prev_req.destination
            else:
                # Only happens if we reach the initial location
                break

        # Reverse path (we reconstructed backwards)
        return path[::-1]

    def _get_subsets(self, items, size):
        """
        Generate all subsets of given size from items.

        Args:
            items: List of items
            size: Desired subset size

        Yields:
            list: Subset of specified size
        """
        if size == 0:
            yield []
        elif size <= len(items):
            for i, item in enumerate(items):
                for subset in self._get_subsets(items[i + 1:], size - 1):
                    yield [item] + subset

    def _optimize_sequence_ilp(self, transporter, requests):
        """
        Optimize request sequence using a focused ILP approach with improved performance.
        Uses a more efficient linearization and warm starting from a greedy solution.
        """
        n = len(requests)
        if n <= 1:
            return requests

        # Create a new ILP model just for sequencing
        from pulp import LpProblem, LpVariable, LpMinimize, lpSum

        seq_model = LpProblem(f"Sequence_Optimization_{transporter.name}", LpMinimize)

        # First, generate a greedy solution to use as warm start
        greedy_sequence = self.sort_requests_by_greedy_chain(transporter, requests)
        greedy_positions = {req.id: p for p, req in enumerate(greedy_sequence)}

        # Create position variables with warm start
        pos_vars = {}
        for req in requests:
            for p in range(n):
                var_name = f"pos_{req.id}_{p}"
                pos_vars[(req.id, p)] = LpVariable(var_name, cat="Binary")
                # Set initial value from greedy solution if available
                if req.id in greedy_positions:
                    if p == greedy_positions[req.id]:
                        pos_vars[(req.id, p)].setInitialValue(1)
                    else:
                        pos_vars[(req.id, p)].setInitialValue(0)

        # Create sequence variables to linearize the product of position variables
        seq_vars = {}

        # Calculate travel times matrix for all pairs of requests and locations
        travel_times = {}
        current_loc = transporter.current_location

        # Precompute all travel times to avoid redundant calculations
        for req in requests:
            # Time from current location to this request
            travel_times[(current_loc, req.origin)] = self.estimate_point_to_point_time(
                current_loc, req.origin)

            # Time to complete this request
            travel_times[(req.origin, req.destination)] = self.estimate_point_to_point_time(
                req.origin, req.destination)

            # Times between requests
            for req2 in requests:
                if req != req2:
                    travel_times[(req.destination, req2.origin)] = self.estimate_point_to_point_time(
                        req.destination, req2.origin)

        # Each request is assigned exactly one position
        for req in requests:
            seq_model += (
                lpSum(pos_vars[(req.id, p)] for p in range(n)) == 1,
                f"ReqPosition_{req.id}"
            )

        # Each position has exactly one request
        for p in range(n):
            seq_model += (
                lpSum(pos_vars[(req.id, p)] for req in requests) == 1,
                f"PosAssigned_{p}"
            )

        # Initialize objective components
        objective_terms = []

        # First request time (from current location)
        for req in requests:
            first_req_time = (
                    travel_times[(current_loc, req.origin)] +
                    travel_times[(req.origin, req.destination)]
            )
            objective_terms.append(pos_vars[(req.id, 0)] * first_req_time)

        # Create sequence variables only for consecutive positions
        # This is more efficient than creating variables for all pairs
        for p in range(n - 1):
            for req1 in requests:
                for req2 in requests:
                    if req1 != req2:
                        # Create sequence variable
                        seq_vars[(req1.id, req2.id, p)] = LpVariable(
                            f"seq_{req1.id}_{req2.id}_{p}", cat="Binary")

                        # Add linearization constraints
                        seq_model += (
                            seq_vars[(req1.id, req2.id, p)] <= pos_vars[(req1.id, p)],
                            f"Seq1_{req1.id}_{req2.id}_{p}"
                        )
                        seq_model += (
                            seq_vars[(req1.id, req2.id, p)] <= pos_vars[(req2.id, p + 1)],
                            f"Seq2_{req1.id}_{req2.id}_{p}"
                        )
                        seq_model += (
                            seq_vars[(req1.id, req2.id, p)] >=
                            pos_vars[(req1.id, p)] + pos_vars[(req2.id, p + 1)] - 1,
                            f"Seq3_{req1.id}_{req2.id}_{p}"
                        )

                        # Add to objective: travel time between consecutive requests
                        transition_time = (
                                travel_times[(req1.destination, req2.origin)] +
                                travel_times[(req2.origin, req2.destination)]
                        )
                        objective_terms.append(seq_vars[(req1.id, req2.id, p)] * transition_time)

        # Set objective to minimize total travel time
        seq_model += lpSum(objective_terms)

        # Solve with increased time limit and CBC settings for better solutions
        solver_options = [
            ("timeLimit", max(10, self.time_limit // 3)),  # More time for better solutions
            ("gapRel", 0.05),  # Accept solutions within 5% of optimal
            ("cuts", "on"),  # Use cutting planes
            ("presolve", "on")  # Use presolve
        ]

        solver = pulp.PULP_CBC_CMD(msg=False, options=[f"{name}={value}" for name, value in solver_options])
        status = seq_model.solve(solver)

        # Extract sequence
        sequence = [None] * n
        for req in requests:
            for p in range(n):
                # Check if variable value is very close to 1 (handle floating point issues)
                var_value = pos_vars[(req.id, p)].value()
                if var_value is not None and abs(var_value - 1.0) < 0.001:
                    sequence[p] = req

        # Fallback to greedy if ILP solution is incomplete
        if None in sequence:
            # Use the greedy sequence we already computed
            return greedy_sequence

        return sequence

    def _generate_warm_start(self):
        """
        Generate a warm start solution using a greedy approach.

        Returns:
            dict: Warm start variable assignments
        """
        warm_start = {}

        # Implement a simple greedy assignment based on nearest transporter
        assigned_requests = set()

        # Sort transporters (could use various metrics)
        sorted_transporters = sorted(self.transporters, key=lambda t: -1)  # Default ordering

        for t in sorted_transporters:
            t_requests = []
            current_loc = t.current_location

            # Sort unassigned requests by distance from current location
            sorted_requests = sorted(
                [r for r in self.requests if r.id not in assigned_requests],
                key=lambda r: self.estimate_point_to_point_time(current_loc, r.origin)
            )

            # Assign closest requests to this transporter
            for r in sorted_requests:
                # Limit number of requests per transporter (could be dynamic)
                max_per_transporter = max(len(self.requests) // len(self.transporters), 1)
                if len(t_requests) >= max_per_transporter:
                    break

                t_requests.append(r)
                assigned_requests.add(r.id)
                current_loc = r.destination  # Update location for next assignment

                # Mark this assignment in warm start
                warm_start[(t.name, r.id)] = 1

        # Mark unassigned pairs as 0
        for t in self.transporters:
            for r in self.requests:
                if (t.name, r.id) not in warm_start:
                    warm_start[(t.name, r.id)] = 0

        return warm_start

    def extract_assignments(self):
        """
        Extract the assignment plan from the optimization results.

        Returns:
            dict: Assignment plan mapping transporter names to lists of requests
        """
        plan = {t.name: [] for t in self.transporters}

        for (t_name, r_id), var in self.assign_vars.items():
            if var.value() == 1:
                req = next(r for r in self.requests if r.id == r_id)
                plan[t_name].append(req)

        # Sort assignments per transporter by travel time from current location
        for t in self.transporters:
            plan[t.name] = self.sort_requests_by_greedy_chain(t, plan[t.name])

        return plan

    def sort_requests_by_greedy_chain(self, transporter, requests):
        """
        Sort requests in a greedy chain to minimize travel time.
        This enhanced version improves the greedy algorithm to find better sequences.

        Args:
            transporter: Transporter object
            requests: List of request objects

        Returns:
            list: Sorted list of requests
        """
        if not requests:
            return []

        from copy import deepcopy
        import heapq

        # Make a copy of the requests to avoid modifying the original
        remaining = deepcopy(requests)
        ordered = []
        current_location = transporter.current_location

        # Try multiple starting points and keep the best sequence
        best_sequence = []
        best_total_time = float('inf')

        # Number of starting points to try (limit to reasonable number for performance)
        num_candidates = min(len(remaining), 5)

        # Sort requests by distance from current location
        candidates = sorted(
            remaining,
            key=lambda r: self.estimate_point_to_point_time(current_location, r.origin)
        )[:num_candidates]

        # Try different starting points
        for start_req in candidates:
            # Make a copy of the remaining requests
            curr_remaining = [r for r in remaining if r != start_req]
            curr_ordered = [start_req]
            curr_location = start_req.destination
            curr_total_time = self.estimate_point_to_point_time(current_location, start_req.origin) + \
                              self.estimate_point_to_point_time(start_req.origin, start_req.destination)

            # Build the rest of the sequence greedily
            while curr_remaining:
                # Look ahead more than just the next request
                # Consider the impact of the next 2 requests if possible
                best_next = None
                best_next_time = float('inf')

                for next_req in curr_remaining:
                    # Time to go to next request
                    time_to_next = self.estimate_point_to_point_time(curr_location, next_req.origin) + \
                                   self.estimate_point_to_point_time(next_req.origin, next_req.destination)

                    # Try to look ahead to the following request
                    min_second_time = float('inf')
                    if len(curr_remaining) > 1:
                        for second_req in curr_remaining:
                            if second_req != next_req:
                                # Time from next_req to second_req
                                second_time = self.estimate_point_to_point_time(next_req.destination,
                                                                                second_req.origin) + \
                                              self.estimate_point_to_point_time(second_req.origin,
                                                                                second_req.destination)
                                min_second_time = min(min_second_time, second_time)

                    # If we can't look ahead, just use the time to next request
                    if min_second_time == float('inf'):
                        min_second_time = 0

                    # Combined score (weighted to prioritize next request but consider look-ahead)
                    combined_time = time_to_next + 0.5 * min_second_time

                    if combined_time < best_next_time:
                        best_next_time = combined_time
                        best_next = next_req

                if best_next:
                    curr_ordered.append(best_next)
                    curr_remaining.remove(best_next)
                    curr_total_time += self.estimate_point_to_point_time(curr_location, best_next.origin) + \
                                       self.estimate_point_to_point_time(best_next.origin, best_next.destination)
                    curr_location = best_next.destination
                else:
                    break

            # Check if this sequence is better than our best so far
            if curr_total_time < best_total_time:
                best_total_time = curr_total_time
                best_sequence = curr_ordered

        return best_sequence

    def estimate_travel_time(self, transporter, request):
        """
        Estimate total travel time for a transporter to complete a request.

        Args:
            transporter: Transporter object
            request: Request object

        Returns:
            float: Estimated travel time
        """
        path_to_origin, _ = transporter.pathfinder.dijkstra(transporter.current_location, request.origin)
        to_origin_time = sum(
            self.graph.get_edge_weight(path_to_origin[i], path_to_origin[i + 1])
            for i in range(len(path_to_origin) - 1)
        )

        path_to_dest, _ = transporter.pathfinder.dijkstra(request.origin, request.destination)
        to_dest_time = sum(
            self.graph.get_edge_weight(path_to_dest[i], path_to_dest[i + 1])
            for i in range(len(path_to_dest) - 1)
        )

        return to_origin_time + to_dest_time

    def estimate_point_to_point_time(self, start, end):
        """
        Estimate travel time between two points.

        Args:
            start: Starting location
            end: Ending location

        Returns:
            float: Estimated travel time
        """
        try:
            path, _ = self.transporters[0].pathfinder.dijkstra(start, end)
            return sum(self.graph.get_edge_weight(path[i], path[i + 1]) for i in range(len(path) - 1))
        except (IndexError, AttributeError, ValueError):
            # Fallback if pathfinding fails
            return 10  # Default time estimate