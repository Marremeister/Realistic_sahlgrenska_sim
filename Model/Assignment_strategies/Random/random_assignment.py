import random


class RandomAssignment:
    def __init__(self, transporters, transport_requests, graph):
        """
        Initialize the random assignment strategy.

        Args:
            transporters: List of available patient transporters
            transport_requests: List of transport requests
            graph: Hospital graph for calculating distances
        """
        self.transporters = transporters
        self.transport_requests = transport_requests
        self.graph = graph

    def generate_assignment_plan(self, transporters, requests):
        """
        Generate an assignment plan ensuring all transporters are busy.

        Args:
            transporters: List of transporters
            requests: List of transport requests

        Returns:
            dict: Assignment plan mapping transporter names to request lists
        """
        # If no requests, distribute empty lists
        if not requests:
            return {t.name: [] for t in transporters}

        # Separate urgent and non-urgent requests
        urgent_requests = [r for r in requests if r.urgent]
        non_urgent_requests = [r for r in requests if not r.urgent]

        # Initialize assignment plan
        plan = {t.name: [] for t in transporters}

        # Handle urgent requests first
        for urgent_request in urgent_requests:
            # Choose a random transporter for the urgent request
            transporter = random.choice(transporters)
            # Prepend urgent request to the transporter's task list
            plan[transporter.name].insert(0, urgent_request)

        # Shuffle non-urgent requests to ensure complete randomness
        shuffled_requests = non_urgent_requests.copy()
        random.shuffle(shuffled_requests)

        # Distribute non-urgent requests completely randomly
        for i, request in enumerate(shuffled_requests):
            # Use modulo to cycle through transporters
            transporter = transporters[i % len(transporters)]
            plan[transporter.name].append(request)

        return plan

    def estimate_travel_time(self, transporter, request):
        """
        Estimate travel time for a transporter to complete a request.

        Args:
            transporter: Patient transporter
            request: Transport request

        Returns:
            float: Estimated travel time in seconds
        """
        try:
            pathfinder = transporter.pathfinder
            graph = transporter.hospital.get_graph()

            # Time from current location to request origin
            path_to_origin, _ = pathfinder.dijkstra(transporter.current_location, request.origin)
            time_to_origin = sum(
                graph.get_edge_weight(path_to_origin[i], path_to_origin[i + 1])
                for i in range(len(path_to_origin) - 1)
            )

            # Time from origin to destination
            path_to_destination, _ = pathfinder.dijkstra(request.origin, request.destination)
            time_to_destination = sum(
                graph.get_edge_weight(path_to_destination[i], path_to_destination[i + 1])
                for i in range(len(path_to_destination) - 1)
            )

            return time_to_origin + time_to_destination

        except Exception:
            # Fallback if path can't be found
            return 9999  # Large penalty time


import random