from pulp import lpSum, LpVariable
from Model.Assignment_strategies.ILP.ilp_core import ILPCore


class ILPEqualWorkload(ILPCore):
    def define_objective(self):
        """
        Define the equal workload objective function.
        This minimizes the maximum number of requests per transporter.
        """
        self.max_requests = LpVariable("max_requests", lowBound=0)

        for t in self.transporters:
            # Count total requests assigned to this transporter
            total = lpSum(
                self.assign_vars[(t.name, r.id)] for r in self.requests
            )
            self.model += (total <= self.max_requests, f"MaxRequests_{t.name}")

        # Set objective to minimize maximum requests per transporter
        self.model += self.max_requests