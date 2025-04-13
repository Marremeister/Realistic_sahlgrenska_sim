import time
import eventlet


class SimulationClock:
    def __init__(self, speed_factor=10):
        """
        Initialize the simulation clock.

        Args:
            speed_factor (float): How much faster simulation time runs compared to real time
        """
        self.start_time_real = time.time()
        self.speed_factor = speed_factor
        self._current_sim_time = 0
        self.running = False
        self._manual_offset = 0  # Allows manually setting time

    def start(self):
        """Start the clock if not already running."""
        if not self.running:
            self.running = True
            eventlet.spawn_n(self._run_clock)

    def _run_clock(self):
        """Background thread to update simulation time."""
        while self.running:
            now_real = time.time()
            elapsed_real = now_real - self.start_time_real
            self._current_sim_time = (elapsed_real * self.speed_factor) + self._manual_offset
            eventlet.sleep(0.1)

    def get_time(self):
        """
        Get the current simulation time.

        Returns:
            float: Current simulation time in seconds
        """
        return self._current_sim_time

    def stop(self):
        """Stop the clock."""
        self.running = False

    def set_time(self, hours=0, minutes=0, seconds=0):
        """
        Manually set the simulation time.

        Args:
            hours (int): Hours to set (0-23)
            minutes (int): Minutes to set (0-59)
            seconds (int): Seconds to set (0-59)
        """
        # Convert time to total seconds
        total_seconds = (hours * 3600) + (minutes * 60) + seconds

        # Set the manual offset
        self._manual_offset = total_seconds

    def set_time_from_timestamp(self, timestamp):
        """
        Set simulation time from a timestamp.

        Args:
            timestamp (str): Timestamp in format 'HH:MM:SS'
        """
        # Parse timestamp
        try:
            hours, minutes, seconds = map(int, timestamp.split(':'))
            self.set_time(hours, minutes, seconds)
        except (ValueError, TypeError):
            raise ValueError("Invalid timestamp format. Use 'HH:MM:SS'")

    def advance_time(self, hours=0, minutes=0, seconds=0):
        """
        Advance the simulation time by a specified amount.

        Args:
            hours (int): Hours to advance
            minutes (int): Minutes to advance
            seconds (int): Seconds to advance
        """
        # Calculate additional seconds
        total_additional_seconds = (hours * 3600) + (minutes * 60) + seconds

        # Add to the manual offset
        self._manual_offset += total_additional_seconds

    def reset_time(self):
        """
        Reset the simulation time to 0 and reset the manual offset.
        """
        self._manual_offset = 0