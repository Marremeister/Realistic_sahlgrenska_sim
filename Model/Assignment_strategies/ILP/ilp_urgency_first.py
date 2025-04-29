from pulp import lpSum
from Model.Assignment_strategies.ILP.ilp_core import ILPCore


class ILPUrgencyFirst(ILPCore):
    def define_objective(self):
        """
        Define the urgency-first objective function.
        This minimizes a weighted sum where urgent requests have higher priority.
        """
        # For urgency-first, we minimize the sum of urgency weights
        # Urgent requests have higher weights (negative contributions)
        # Non-urgent requests have lower weights (positive contributions)
        self.model += lpSum(
            self.assign_vars[(t.name, r.id)] * (-10 if r.urgent else 1)
            for t in self.transporters
            for r in self.requests
        )