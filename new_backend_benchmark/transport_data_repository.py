import os
import json
import logging
import random
from datetime import datetime


class TransportDataRepository:
    """
    Repository class that loads pre-analyzed transport data from files
    and provides methods to generate realistic benchmark requests.
    """

    def __init__(self, data_dir='analysis_output'):
        """
        Initialize the repository with the directory containing analysis data.

        Args:
            data_dir (str): Directory containing analysis data files
        """
        self.data_dir = data_dir
        self.logger = self._setup_logger()

        # Data containers
        self.hourly_rates = {}
        self.time_range_stats = {}
        self.od_pairs_by_time = {}
        self.transport_types = {}
        self.urgency_distribution = {}
        self.departments = []

        # Load data if directory exists
        if os.path.exists(data_dir):
            self.load_data()
        else:
            self.logger.warning(f"Data directory {data_dir} does not exist. Repository not initialized.")

    def _setup_logger(self):
        """Set up a logger for the TransportDataRepository."""
        logger = logging.getLogger("TransportDataRepository")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self):
        """
        Load all data from files in the data directory.

        Returns:
            bool: True if data was loaded successfully
        """
        try:
            # Load hourly rates
            hourly_rates_file = os.path.join(self.data_dir, 'hourly_request_stats.json')
            if os.path.exists(hourly_rates_file):
                with open(hourly_rates_file, 'r') as f:
                    self.hourly_rates = json.load(f)
                self.logger.info(f"Loaded hourly rates from {hourly_rates_file}")

            # Load time range statistics
            time_range_file = os.path.join(self.data_dir, 'time_range_stats.json')
            if os.path.exists(time_range_file):
                with open(time_range_file, 'r') as f:
                    self.time_range_stats = json.load(f)
                self.logger.info(f"Loaded time range statistics from {time_range_file}")

            # Load origin-destination pairs by time
            od_pairs_file = os.path.join(self.data_dir, 'od_pairs_by_time.json')
            if os.path.exists(od_pairs_file):
                with open(od_pairs_file, 'r') as f:
                    self.od_pairs_by_time = json.load(f)
                self.logger.info(f"Loaded origin-destination pairs from {od_pairs_file}")

            # Load transport types (default if file doesn't exist)
            self.transport_types = {
                'stretcher': 0.5,
                'wheelchair': 0.3,
                'bed': 0.2
            }

            # Load urgency distribution (default if file doesn't exist)
            self.urgency_distribution = {
                'true': 0.2,
                'false': 0.8
            }

            # Load departments (all unique departments from OD pairs)
            self.departments = self._extract_unique_departments()

            self.logger.info(f"Successfully loaded data from {self.data_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def _extract_unique_departments(self):
        """
        Extract all unique departments from the loaded OD pairs.

        Returns:
            list: List of unique department names
        """
        departments = set()

        # Extract from all time ranges
        for time_range, pairs in self.od_pairs_by_time.items():
            for pair in pairs:
                departments.add(pair['origin'])
                departments.add(pair['destination'])

        return list(departments)

    def get_request_rate(self, start_hour, end_hour):
        """
        Get the request rate for a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            float: Average requests per hour for the time range
        """
        # Try to find exact time range
        time_key = f"{start_hour:02d}-{end_hour:02d}"

        # Check in time_range_stats first
        if time_key in self.time_range_stats:
            return self.time_range_stats[time_key].get('requests_per_hour', 0)

        # If not found, calculate average from hourly rates
        total_rate = 0
        count = 0

        current_hour = start_hour
        while current_hour != end_hour:
            next_hour = (current_hour + 1) % 24
            hourly_key = f"{current_hour:02d}-{next_hour:02d}"

            if hourly_key in self.hourly_rates:
                hour_rate = self.hourly_rates[hourly_key].get('requests_per_hour', 0)
                total_rate += hour_rate
                count += 1

            current_hour = next_hour

        return total_rate / max(1, count)  # Avoid division by zero

    def _find_best_time_range(self, start_hour, end_hour):
        """
        Find the best matching predefined time range for the given hours.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            str: Best matching time range key
        """
        # First check if exact range exists
        exact_key = f"{start_hour:02d}-{end_hour:02d}"
        if exact_key in self.od_pairs_by_time:
            return exact_key

        # If not, find most overlapping range
        predefined_ranges = []
        for range_key in self.od_pairs_by_time.keys():
            range_start, range_end = map(int, range_key.split('-'))
            predefined_ranges.append((range_start, range_end, range_key))

        # If no ranges, return None
        if not predefined_ranges:
            return None

        # Calculate hours in requested range
        if start_hour <= end_hour:
            requested_hours = set(range(start_hour, end_hour))
        else:
            requested_hours = set(list(range(start_hour, 24)) + list(range(0, end_hour)))

        best_overlap = 0
        best_range = predefined_ranges[0][2]  # Default to first range

        for range_start, range_end, range_key in predefined_ranges:
            # Calculate hours in predefined range
            if range_start <= range_end:
                range_hours = set(range(range_start, range_end))
            else:
                range_hours = set(list(range(range_start, 24)) + list(range(0, range_end)))

            # Calculate overlap
            overlap = len(requested_hours.intersection(range_hours))

            if overlap > best_overlap:
                best_overlap = overlap
                best_range = range_key

        return best_range

    def generate_benchmark_requests(self, start_hour, end_hour, num_requests=None):
        """
        Generate realistic transport requests for benchmarking based on time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            num_requests (int, optional): Number of requests to generate.
                                         If None, calculated based on average rate.

        Returns:
            list: List of (origin, destination, transport_type, urgent) tuples
        """
        # Calculate default number of requests if not specified
        if num_requests is None:
            hourly_rate = self.get_request_rate(start_hour, end_hour)

            # Calculate hours in range
            if end_hour >= start_hour:
                hours = end_hour - start_hour
            else:
                hours = (24 - start_hour) + end_hour

            num_requests = int(hourly_rate * hours)
            # Ensure at least one request
            num_requests = max(1, num_requests)

        # Find best matching time range for OD pair distribution
        best_range = self._find_best_time_range(start_hour, end_hour)

        if not best_range or best_range not in self.od_pairs_by_time:
            self.logger.warning(
                f"No origin-destination data for time range {start_hour}-{end_hour}. Using random pairs.")
            # Fallback: Generate random OD pairs from known departments
            return self._generate_random_requests(num_requests)

        # Get OD pairs with probabilities
        od_pairs = self.od_pairs_by_time[best_range]

        # Convert to format needed for random.choices
        pairs = []
        weights = []

        for pair in od_pairs:
            pairs.append((pair['origin'], pair['destination']))
            weights.append(pair['probability'])

        # Normalize weights if needed
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]

        # Generate requests
        requests = []
        for _ in range(num_requests):
            if not pairs:  # Safety check
                break

            # Randomly select O-D pair based on probability distribution
            origin, dest = random.choices(pairs, weights=weights, k=1)[0]

            # Randomly select transport type
            transport_types = list(self.transport_types.keys())
            transport_weights = list(self.transport_types.values())
            transport_type = random.choices(transport_types, weights=transport_weights, k=1)[0]

            # Randomly select urgency
            urgencies = [True, False]  # Convert string keys to boolean
            urgency_weights = [self.urgency_distribution.get('true', 0.2),
                               self.urgency_distribution.get('false', 0.8)]
            urgent = random.choices(urgencies, weights=urgency_weights, k=1)[0]

            requests.append((origin, dest, transport_type, urgent))

        return requests

    def _generate_random_requests(self, num_requests):
        """
        Generate random transport requests when no time-specific data is available.

        Args:
            num_requests (int): Number of requests to generate

        Returns:
            list: List of (origin, destination, transport_type, urgent) tuples
        """
        # Ensure we have departments
        if not self.departments or len(self.departments) < 2:
            self.logger.warning("No departments found for random request generation.")
            return []

        # Generate random OD pairs
        requests = []
        for _ in range(num_requests):
            # Pick random origin and destination (ensuring they're different)
            origin, dest = random.sample(self.departments, 2)

            # Randomly select transport type
            transport_types = list(self.transport_types.keys())
            transport_weights = list(self.transport_types.values())
            transport_type = random.choices(transport_types, weights=transport_weights, k=1)[0]

            # Randomly select urgency
            urgencies = [True, False]
            urgency_weights = [self.urgency_distribution.get('true', 0.2),
                               self.urgency_distribution.get('false', 0.8)]
            urgent = random.choices(urgencies, weights=urgency_weights, k=1)[0]

            requests.append((origin, dest, transport_type, urgent))

        return requests

    def get_available_time_ranges(self):
        """
        Get all available predefined time ranges in the data.

        Returns:
            list: List of time range strings (e.g., ["08-12", "12-17"])
        """
        return list(self.od_pairs_by_time.keys())

    def get_hourly_rates_for_chart(self):
        """
        Get hourly rates in a format suitable for charting.

        Returns:
            dict: Dictionary with labels and data arrays for charting
        """
        hours = []
        rates = []

        # Sort by hour
        for i in range(24):
            next_hour = (i + 1) % 24
            key = f"{i:02d}-{next_hour:02d}"

            if key in self.hourly_rates:
                hours.append(f"{i:02d}:00")
                rates.append(self.hourly_rates[key].get('requests_per_hour', 0))
            else:
                hours.append(f"{i:02d}:00")
                rates.append(0)

        return {
            'labels': hours,
            'data': rates
        }

    @classmethod
    def check_data_exists(cls, data_dir='analysis_output'):
        """
        Check if analysis data exists in the specified directory.

        Args:
            data_dir (str): Directory to check for analysis data

        Returns:
            bool: True if essential data files exist
        """
        # Check for essential files
        required_files = [
            'hourly_request_stats.json',
            'od_pairs_by_time.json'
        ]

        if not os.path.exists(data_dir):
            return False

        return all(os.path.exists(os.path.join(data_dir, file)) for file in required_files)


# Ensure TransportDataAnalyzer can export all required data
def enhance_transport_data_analyzer_export(analyzer, output_dir='analysis_output'):
    """
    Ensure the TransportDataAnalyzer exports all data required by the repository.
    This function complements the existing export_time_based_statistics method.

    Args:
        analyzer: TransportDataAnalyzer instance with loaded and cleaned data
        output_dir: Directory to save the data

    Returns:
        bool: True if successful
    """
    try:
        # Call the main export method (already implemented)
        analyzer.export_time_based_statistics(output_dir)

        # Additional exports if needed

        # Export transport type distribution
        type_col = analyzer._find_column_containing('type', analyzer.cleaned_data.columns)
        if type_col:
            type_counts = analyzer.cleaned_data[type_col].value_counts()
            type_dist = (type_counts / len(analyzer.cleaned_data)).to_dict()

            with open(os.path.join(output_dir, 'transport_types.json'), 'w') as f:
                json.dump(type_dist, f, indent=2)

        # Export urgency distribution
        urgency_col = analyzer._find_column_containing('urgent', analyzer.cleaned_data.columns)
        if urgency_col:
            urgency_counts = analyzer.cleaned_data[urgency_col].value_counts()
            urgency_dist = (urgency_counts / len(analyzer.cleaned_data)).to_dict()

            with open(os.path.join(output_dir, 'urgency_distribution.json'), 'w') as f:
                json.dump(urgency_dist, f, indent=2)

        # Export all departments
        departments = analyzer.get_all_departments()

        with open(os.path.join(output_dir, 'departments.json'), 'w') as f:
            json.dump(departments, f, indent=2)

        # Export metadata (creation timestamp, data source, etc.)
        metadata = {
            'created_at': datetime.now().isoformat(),
            'data_source': getattr(analyzer, 'file_path', 'unknown'),
            'record_count': len(analyzer.cleaned_data),
            'department_count': len(departments),
            'version': '1.0'
        }

        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        logging.error(f"Error enhancing export: {str(e)}")
        return False