from pulp import lpSum, LpVariable
from Model.Assignment_strategies.ILP.ilp_core import ILPCore


class ILPMakespan(ILPCore):
    """
    Enhanced ILP Makespan optimization with better look-ahead capabilities.
    Balances makespan with total travel time for improved overall performance.
    """

    def __init__(self, transporters, requests, graph, use_two_phase=True,
                 time_limit=30, use_warm_start=True, balanced_objective=True):
        """
        Initialize the ILP Makespan optimizer.

        Args:
            transporters: List of transporter objects
            requests: List of transport request objects
            graph: Hospital graph object
            use_two_phase: Whether to use the two-phase approach
            time_limit: Maximum time in seconds for optimization
            use_warm_start: Whether to use warm starting for the solver
            balanced_objective: Whether to use a balanced objective function
        """
        super().__init__(transporters, requests, graph, use_two_phase, time_limit, use_warm_start)
        self.balanced_objective = balanced_objective

    def define_objective(self):
        """
        Define the makespan objective function with improved performance.
        This uses a balanced approach that minimizes both makespan and total travel time.
        """
        self.makespan = LpVariable("makespan", lowBound=0)

        # For small problems, we can use a modified standard approach
        if not self.use_two_phase or len(self.requests) <= self.two_phase_threshold:
            # Additional variable for total travel time
            self.total_travel_time = LpVariable("total_travel_time", lowBound=0)

            # Constraints for each transporter
            for t in self.transporters:
                # Constraint for makespan (max completion time)
                transporter_time = lpSum(
                    self.assign_vars[(t.name, r.id)] * self.estimate_travel_time(t, r)
                    for r in self.requests
                )
                self.model += (transporter_time <= self.makespan, f"MakespanLimit_{t.name}")

            # Constraint for total travel time
            self.model += (
                lpSum(
                    self.assign_vars[(t.name, r.id)] * self.estimate_travel_time(t, r)
                    for t in self.transporters for r in self.requests
                ) <= self.total_travel_time,
                "TotalTravelTime"
            )

            # Use balanced objective if enabled, otherwise use pure makespan
            if self.balanced_objective:
                # Use a weighted sum of makespan and total travel time
                # The weight factors prioritize makespan but also consider total time
                avg_request_time = sum(
                    self.estimate_travel_time(t, r)
                    for t in self.transporters for r in self.requests
                ) / (max(1, len(self.transporters) * len(self.requests)))

                # Scale factors to make the objectives comparable
                makespan_weight = 1.0
                total_time_weight = 0.2 / max(1, len(self.requests))

                self.model += (makespan_weight * self.makespan +
                               total_time_weight * self.total_travel_time)
            else:
                # Original pure makespan objective
                self.model += self.makespan
        else:
            # For the two-phase approach, we need to create a better approximation
            # of sequence-dependent travel times for Phase 1

            # Create proximity scores for pairs of requests
            proximity_scores = {}
            for r1 in self.requests:
                for r2 in self.requests:
                    if r1 != r2:
                        # Lower score means these requests should be assigned to the same transporter
                        # Distance from r1's destination to r2's origin
                        proximity_scores[(r1.id, r2.id)] = self.estimate_point_to_point_time(
                            r1.destination, r2.origin)

            # Calculate base travel times for each transporter-request pair
            base_times = {}
            for t in self.transporters:
                for r in self.requests:
                    base_times[(t.name, r.id)] = self.estimate_travel_time(t, r)

            # Create assignment groups to encourage related requests to be assigned together
            for t in self.transporters:
                # For each request, create constraints that encourage assigning related requests
                # to the same transporter
                for r1 in self.requests:
                    # Find the top 3 closest requests (by destination->origin proximity)
                    closest_requests = sorted(
                        [r2 for r2 in self.requests if r2 != r1],
                        key=lambda r2: proximity_scores.get((r1.id, r2.id), float('inf'))
                    )[:3]

                    # Add a soft constraint to assign close requests to the same transporter
                    for r2 in closest_requests:
                        if proximity_scores.get((r1.id, r2.id), float('inf')) < 50:  # Threshold for "close" requests
                            # This creates a correlation between assigning r1 and r2 to the same transporter
                            # But doesn't force it as a hard constraint
                            proximity_penalty = proximity_scores.get((r1.id, r2.id), 0) / 10

                            # Add a small penalty to the makespan if only one of the requests is assigned
                            # to this transporter (encourages keeping them together)
                            self.model += (
                                proximity_penalty * (
                                        self.assign_vars[(t.name, r1.id)] -
                                        self.assign_vars[(t.name, r2.id)]
                                ) <= self.makespan,
                                f"Proximity_{t.name}_{r1.id}_{r2.id}_1"
                            )
                            self.model += (
                                proximity_penalty * (
                                        self.assign_vars[(t.name, r2.id)] -
                                        self.assign_vars[(t.name, r1.id)]
                                ) <= self.makespan,
                                f"Proximity_{t.name}_{r1.id}_{r2.id}_2"
                            )

            # Add standard makespan constraints that consider base travel times
            for t in self.transporters:
                # Sum of all assigned request times (simplified approximation)
                total_time = lpSum(
                    self.assign_vars[(t.name, r.id)] * base_times[(t.name, r.id)]
                    for r in self.requests
                )

                # Add potential sequence-dependent overhead based on number of locations
                # This penalizes assigning many scattered requests to one transporter
                unique_locations = {}
                for r in self.requests:
                    # Count contribution to unique origins
                    if r.origin not in unique_locations:
                        unique_locations[r.origin] = LpVariable(
                            f"has_origin_{t.name}_{r.origin}", cat="Binary")
                        self.model += (
                            unique_locations[r.origin] >=
                            lpSum(
                                self.assign_vars[(t.name, r2.id)]
                                for r2 in self.requests if r2.origin == r.origin
                            ) / max(1, sum(1 for r2 in self.requests if r2.origin == r.origin)),
                            f"HasOrigin_{t.name}_{r.origin}"
                        )

                    # Count contribution to unique destinations
                    if r.destination not in unique_locations:
                        unique_locations[r.destination] = LpVariable(
                            f"has_dest_{t.name}_{r.destination}", cat="Binary")
                        self.model += (
                            unique_locations[r.destination] >=
                            lpSum(
                                self.assign_vars[(t.name, r2.id)]
                                for r2 in self.requests if r2.destination == r.destination
                            ) / max(1, sum(1 for r2 in self.requests if r2.destination == r.destination)),
                            f"HasDest_{t.name}_{r.destination}"
                        )

                # Add a location diversity penalty to better account for travel between locations
                locations_penalty = lpSum(unique_locations.values()) * 10

                # Final makespan constraint includes base time and location diversity
                self.model += (
                    total_time + locations_penalty <= self.makespan,
                    f"EnhancedMakespan_{t.name}"
                )

            # Set the objective to minimize makespan
            self.model += self.makespan