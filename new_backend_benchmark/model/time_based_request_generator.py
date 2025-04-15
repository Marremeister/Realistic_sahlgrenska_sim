"""
Time-based request generator for incremental benchmarks.
Adapts existing simulation and repository code to schedule requests over time.
"""

import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from new_backend_benchmark.execution.repository.transport_data_repository import TransportDataRepository


class TimeBasedRequestGenerator:
    """
    Generates and schedules transport requests over time based on real hospital data.

    This class adapts the existing TransportDataRepository and simulation_realistic logic
    to schedule a set of requests incrementally over a time period for benchmarking.
    """

    def __init__(self, requests: List, start_time: float, end_time: float,
                 time_distribution: str = "realistic", data_dir: str = 'analysis_output'):
        """
        Initialize the time-based request generator.

        Args:
            requests: List of request objects to schedule
            start_time: Simulation start time (seconds)
            end_time: Simulation end time (seconds)
            time_distribution: How to distribute requests over time
                - "realistic": Use real hospital data patterns
                - "uniform": Evenly spaced distribution
                - "random": Completely random distribution
            data_dir: Directory containing hospital data files
        """
        self.requests = requests
        self.start_time = start_time
        self.end_time = end_time
        self.time_distribution = time_distribution
        self.scheduled_requests = []
        self.logger = logging.getLogger("TimeBasedRequestGenerator")

        # Use the existing data repository
        self.repository = TransportDataRepository(data_dir)

        # Map simulation time to hospital hours
        self.start_hour = int((start_time / 3600) % 24)
        self.end_hour = int((end_time / 3600) % 24)

        # Initialize the schedule
        self._schedule_requests()

    def _schedule_requests(self):
        """
        Generate a schedule for when each request should appear.

        The distribution will depend on the selected time_distribution strategy.
        All scheduled requests are sorted by time for efficient processing.
        """
        if not self.requests:
            self.logger.warning("No requests to schedule")
            return

        time_range = self.end_time - self.start_time

        if time_range <= 0:
            self.logger.error("Invalid time range: end_time must be greater than start_time")
            raise ValueError("End time must be greater than start time")

        # Apply the selected distribution strategy
        if self.time_distribution == "uniform":
            self._apply_uniform_distribution(time_range)
        elif self.time_distribution == "realistic":
            self._apply_realistic_distribution(time_range)
        else:
            # Default to random distribution
            self._apply_random_distribution(time_range)

        # Sort scheduled requests by time
        self.scheduled_requests.sort(key=lambda x: x["scheduled_time"])

        self.logger.info(
            f"Scheduled {len(self.requests)} requests from {self.start_time} to {self.end_time} using {self.time_distribution} distribution")

    def _apply_random_distribution(self, time_range: float):
        """
        Schedule requests randomly across the time range.

        Args:
            time_range: Duration of the simulation in seconds
        """
        for request in self.requests:
            # Generate a random time point within the range
            scheduled_time = self.start_time + random.random() * time_range

            self.scheduled_requests.append({
                "request": request,
                "scheduled_time": scheduled_time,
                "processed": False
            })

    def _apply_uniform_distribution(self, time_range: float):
        """
        Schedule requests evenly across the time range.

        Args:
            time_range: Duration of the simulation in seconds
        """
        request_count = len(self.requests)

        if request_count <= 1:
            # With one request, just put it in the middle
            scheduled_time = self.start_time + (time_range / 2)
            self.scheduled_requests.append({
                "request": self.requests[0],
                "scheduled_time": scheduled_time,
                "processed": False
            })
        else:
            # Calculate time interval between requests
            interval = time_range / (request_count - 1) if request_count > 1 else time_range

            for i, request in enumerate(self.requests):
                scheduled_time = self.start_time + (i * interval)
                self.scheduled_requests.append({
                    "request": request,
                    "scheduled_time": scheduled_time,
                    "processed": False
                })

    def _apply_realistic_distribution(self, time_range: float):
        """
        Schedule requests using the real hospital data patterns.
        Uses TransportDataRepository to get realistic request distributions by hour.

        Args:
            time_range: Duration of the simulation in seconds
        """
        try:
            # Get hourly rate data from repository
            hourly_rates = {}

            # Map simulation hours to real hours
            current_hour = self.start_hour
            while current_hour != self.end_hour:
                next_hour = (current_hour + 1) % 24
                hour_key = f"{current_hour:02d}-{next_hour:02d}"

                # Get the request rate for this hour
                rate = self.repository.get_request_rate(current_hour, next_hour)
                hourly_rates[hour_key] = rate

                current_hour = next_hour

            # If no valid hourly rates, fall back to random
            if not hourly_rates or sum(hourly_rates.values()) == 0:
                self.logger.warning("No valid hourly rates found, falling back to random distribution")
                self._apply_random_distribution(time_range)
                return

            # Create probability weights for each hour
            total_rate = sum(hourly_rates.values())
            hour_probabilities = {hour: rate / total_rate for hour, rate in hourly_rates.items()}

            # Distribute requests according to hourly probabilities
            for request in self.requests:
                # Choose an hour based on probability
                hour_key = random.choices(
                    list(hour_probabilities.keys()),
                    weights=list(hour_probabilities.values()),
                    k=1
                )[0]

                # Extract hour range
                start_h, end_h = map(int, hour_key.split('-'))

                # Convert to simulation time
                hour_start_time = self.start_time
                hour_duration = 3600  # 1 hour in seconds

                # Calculate offset for this hour in the simulation time range
                if self.end_time - self.start_time <= 3600:
                    # If simulation is less than an hour, use the full range
                    hour_offset = 0
                    hour_length = time_range
                else:
                    # Otherwise map to the correct hour in the simulation
                    total_hours = int(time_range / 3600)
                    hour_index = (start_h - self.start_hour) % 24
                    hour_offset = (hour_index % total_hours) * 3600
                    hour_length = min(3600, time_range - hour_offset)

                # Generate a random time within this hour
                relative_time = random.random() * hour_length
                scheduled_time = self.start_time + hour_offset + relative_time

                # Ensure within bounds
                scheduled_time = max(self.start_time, min(self.end_time, scheduled_time))

                # Add to scheduled requests
                self.scheduled_requests.append({
                    "request": request,
                    "scheduled_time": scheduled_time,
                    "processed": False,
                    "hour": hour_key  # Store for analysis
                })

        except Exception as e:
            # Fall back to random distribution if anything goes wrong
            self.logger.warning(f"Error applying realistic distribution: {str(e)}. Falling back to random.")
            self._apply_random_distribution(time_range)

    def get_active_requests(self, current_time: float) -> List:
        """
        Get requests that should be active at the current time.

        Args:
            current_time: Current simulation time in seconds

        Returns:
            List of request objects that should be active
        """
        active_requests = []

        for scheduled_req in self.scheduled_requests:
            if (scheduled_req["scheduled_time"] <= current_time and
                    not scheduled_req["processed"]):
                active_requests.append(scheduled_req["request"])
                scheduled_req["processed"] = True

        return active_requests

    def get_next_request_time(self) -> Optional[float]:
        """
        Get the scheduled time of the next unprocessed request.

        Returns:
            Float: Time of the next request or None if all processed
        """
        for scheduled_req in self.scheduled_requests:
            if not scheduled_req["processed"]:
                return scheduled_req["scheduled_time"]
        return None

    def all_processed(self) -> bool:
        """
        Check if all requests have been processed.

        Returns:
            Bool: True if all requests have been processed
        """
        return all(req["processed"] for req in self.scheduled_requests)

    def get_completion_percentage(self) -> float:
        """
        Get the percentage of requests that have been processed.

        Returns:
            Float: Percentage of completion (0-100)
        """
        if not self.scheduled_requests:
            return 100.0

        processed_count = sum(1 for req in self.scheduled_requests if req["processed"])
        return (processed_count / len(self.scheduled_requests)) * 100

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the request schedule.

        Returns:
            Dict with schedule metrics
        """
        if not self.scheduled_requests:
            return {
                "total_requests": 0,
                "processed_requests": 0,
                "completion_percentage": 0,
                "time_range": self.end_time - self.start_time
            }

        processed_count = sum(1 for req in self.scheduled_requests if req["processed"])

        # Get distribution of requests by hour
        hour_distribution = {}
        for req in self.scheduled_requests:
            hour = req.get("hour", "unknown")
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1

        return {
            "total_requests": len(self.scheduled_requests),
            "processed_requests": processed_count,
            "completion_percentage": (processed_count / len(self.scheduled_requests)) * 100,
            "time_range": self.end_time - self.start_time,
            "distribution_type": self.time_distribution,
            "hour_distribution": hour_distribution
        }

    def get_hourly_distribution_data(self) -> Dict[str, Any]:
        """
        Get data about hourly request distribution for visualization.

        Returns:
            Dict with hourly rates and distribution data
        """
        # Get hourly rates chart data from repository
        hourly_rates = self.repository.get_hourly_rates_for_chart()

        # Get actual distribution from our scheduled requests
        hours = [f"{i:02d}:00" for i in range(24)]
        scheduled_counts = [0] * 24

        for req in self.scheduled_requests:
            hour_key = req.get("hour", "")
            if hour_key and '-' in hour_key:
                hour = int(hour_key.split('-')[0])
                scheduled_counts[hour] += 1

        return {
            "hourly_rates": hourly_rates,
            "schedule_distribution": {
                "labels": hours,
                "data": scheduled_counts
            }
        }