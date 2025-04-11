# Model/Data_processor/time_based_transport_analyzer.py
"""
Specialized analyzer for time-based transport data statistics.
Creates detailed JSON files with hourly origin-destination patterns.
"""
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from Model.Data_processor.transport_data_analyzer import TransportDataAnalyzer
from Model.Data_processor.department_name_normalizer import DepartmentNameNormalizer


class TimeBasedTransportAnalyzer:
    """
    Analyzes transport data to extract hourly patterns and generate
    detailed statistics for origin-destination pairs by hour of day.
    Builds on the base TransportDataAnalyzer but with focus on time patterns.
    """

    def __init__(self, file_path, output_dir='analysis_output'):
        """
        Initialize the time-based transport analyzer.

        Args:
            file_path (str): Path to the CSV or Excel file with transport data
            output_dir (str): Directory to save analysis output
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.base_analyzer = TransportDataAnalyzer(file_path)
        self.hourly_stats = defaultdict(dict)
        self.od_hourly_frequencies = defaultdict(lambda: defaultdict(int))
        self.hourly_request_count = defaultdict(int)
        self.transport_type_hourly = defaultdict(lambda: defaultdict(int))
        self.urgency_hourly = defaultdict(lambda: defaultdict(int))
        self.logger = self._setup_logger()
        self.name_mapping = {}  # Will be populated during normalization

    def _setup_logger(self):
        """Set up a logger for the TimeBasedTransportAnalyzer."""
        logger = logging.getLogger("TimeBasedTransportAnalyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_and_prepare_data(self):
        """
        Load and prepare data using the base analyzer.
        Extends with additional time-based preparation.

        Returns:
            bool: True if successful, False otherwise
        """
        # Use the base analyzer to load and clean data
        if not self.base_analyzer.load_data():
            self.logger.error("Failed to load data.")
            return False

        cleaned_data = self.base_analyzer.clean_data()
        if cleaned_data is None or cleaned_data.empty:
            self.logger.error("Failed to clean data.")
            return False

        # Get column mapping from base analyzer
        if hasattr(self.base_analyzer, '_column_mapping'):
            self.column_mapping = self.base_analyzer._column_mapping
        else:
            self.column_mapping = {
                'origin': self._find_column_containing('origin', cleaned_data.columns),
                'destination': self._find_column_containing('destination', cleaned_data.columns),
                'start': self._find_column_containing('start', cleaned_data.columns),
                'end': self._find_column_containing('end', cleaned_data.columns),
                'transport_type': self._find_column_containing('type', cleaned_data.columns),
                'urgent': self._find_column_containing('urgent', cleaned_data.columns)
            }

        self.logger.info(f"Column mapping: {self.column_mapping}")

        # Normalize department names
        self._normalize_department_names()

        # Check if the start time column is in datetime format
        start_time_col = self.column_mapping.get('start')
        if start_time_col and start_time_col in cleaned_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(cleaned_data[start_time_col]):
                try:
                    # Try with explicit format and European date format (day first)
                    cleaned_data[start_time_col] = pd.to_datetime(
                        cleaned_data[start_time_col],
                        format='%d-%m-%Y %H:%M:%S',
                        errors='coerce'
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to parse dates with explicit format: {str(e)}")
                    try:
                        # Fallback to automatic format detection with day first
                        cleaned_data[start_time_col] = pd.to_datetime(
                            cleaned_data[start_time_col],
                            dayfirst=True,
                            errors='coerce'
                        )
                    except Exception as e2:
                        self.logger.error(f"Failed to parse dates: {str(e2)}")
                        return False

        self.cleaned_data = cleaned_data
        return True

    def _find_column_containing(self, keyword, columns):
        """Find a column containing the keyword (case-insensitive)."""
        matches = [col for col in columns if keyword.lower() in col.lower()]
        return matches[0] if matches else None

    def _normalize_department_names(self):
        """
        Normalize department names to ensure consistency using the shared normalizer.
        """
        if not hasattr(self.base_analyzer, 'cleaned_data') or self.base_analyzer.cleaned_data is None:
            self.logger.error("No cleaned data available for normalization.")
            return

        origin_col = self.column_mapping.get('origin')
        dest_col = self.column_mapping.get('destination')

        if not origin_col or not dest_col:
            self.logger.error("Could not find origin or destination columns.")
            return

        # Get all unique department names
        all_departments = set()
        if origin_col in self.base_analyzer.cleaned_data.columns:
            all_departments.update(self.base_analyzer.cleaned_data[origin_col].dropna().unique())
        if dest_col in self.base_analyzer.cleaned_data.columns:
            all_departments.update(self.base_analyzer.cleaned_data[dest_col].dropna().unique())

        # Use the shared normalizer
        normalizer = DepartmentNameNormalizer(self.output_dir)  # Use the standard output dir

        # First try to load existing mapping
        normalizer.load_existing_mapping()

        # Then normalize our departments
        self.name_mapping = normalizer.normalize_departments(all_departments)

        # Save the mapping for future use if it was created from scratch
        if not normalizer.load_existing_mapping():
            normalizer.save_mapping('time_based_name_mapping.json')

        self.logger.info(
            f"Normalized {len(all_departments)} department names to {len(set(self.name_mapping.values()))} unique departments")

    # Method removed as it's now handled by the DepartmentNameNormalizer

    def analyze_hourly_patterns(self):
        """
        Analyze hourly patterns in the transport data.

        Returns:
            bool: True if successful, False otherwise
        """
        if not hasattr(self, 'cleaned_data') or self.cleaned_data is None:
            self.logger.error("No cleaned data available for analysis.")
            return False

        start_time_col = self.column_mapping.get('start')
        origin_col = self.column_mapping.get('origin')
        dest_col = self.column_mapping.get('destination')
        type_col = self.column_mapping.get('transport_type')
        urgent_col = self.column_mapping.get('urgent')

        if not start_time_col or not origin_col or not dest_col:
            self.logger.error("Missing required columns for hourly analysis.")
            return False

        self.logger.info("Analyzing hourly transport patterns...")

        # Reset collections
        self.hourly_stats = defaultdict(dict)
        self.od_hourly_frequencies = defaultdict(lambda: defaultdict(int))
        self.hourly_request_count = defaultdict(int)
        self.transport_type_hourly = defaultdict(lambda: defaultdict(int))
        self.urgency_hourly = defaultdict(lambda: defaultdict(int))

        # Extract hour from start time
        if pd.api.types.is_datetime64_any_dtype(self.cleaned_data[start_time_col]):
            self.cleaned_data['hour'] = self.cleaned_data[start_time_col].dt.hour
        else:
            self.logger.error("Start time column is not in datetime format.")
            return False

        # Count requests by hour
        hour_counts = self.cleaned_data['hour'].value_counts()
        for hour, count in hour_counts.items():
            self.hourly_request_count[int(hour)] = int(count)

        # Count origin-destination pairs by hour
        for hour in range(24):
            hour_data = self.cleaned_data[self.cleaned_data['hour'] == hour]

            # Skip if no data for this hour
            if hour_data.empty:
                continue

            # Count OD pairs
            od_counts = hour_data.groupby([origin_col, dest_col]).size()
            total_hour_requests = len(hour_data)

            for (origin, dest), count in od_counts.items():
                # Normalize department names
                norm_origin = self.name_mapping.get(origin, origin)
                norm_dest = self.name_mapping.get(dest, dest)

                # Skip if origin and destination are the same after normalization
                if norm_origin == norm_dest:
                    continue

                # Store frequency
                od_key = f"{norm_origin}→{norm_dest}"
                self.od_hourly_frequencies[hour][od_key] = count

            # Count transport types if available
            if type_col and type_col in hour_data.columns:
                type_counts = hour_data[type_col].value_counts()
                for t_type, count in type_counts.items():
                    self.transport_type_hourly[hour][str(t_type)] = int(count)

            # Count urgency if available
            if urgent_col and urgent_col in hour_data.columns:
                urgency_counts = hour_data[urgent_col].value_counts()
                for urgent, count in urgency_counts.items():
                    # Convert to boolean string for consistency
                    urgent_key = 'true' if pd.notna(urgent) and bool(urgent) else 'false'
                    self.urgency_hourly[hour][urgent_key] = int(count)

        # Calculate hourly statistics
        self._calculate_hourly_statistics()

        self.logger.info("Hourly pattern analysis complete.")
        return True

    def _calculate_hourly_statistics(self):
        """Calculate statistics for each hour."""
        for hour in range(24):
            # Request count and rate (per hour)
            request_count = self.hourly_request_count.get(hour, 0)

            # Create stats dictionary
            hour_key = f"{hour:02d}-{(hour + 1) % 24:02d}"
            self.hourly_stats[hour_key] = {
                'request_count': request_count,
                'requests_per_hour': request_count,  # Same for hourly stats
                'od_pair_count': len(self.od_hourly_frequencies.get(hour, {})),
                'transport_types': dict(self.transport_type_hourly.get(hour, {})),
                'urgency_distribution': dict(self.urgency_hourly.get(hour, {})),
            }

            # Calculate OD pair probabilities
            od_pairs = []
            total_requests = request_count

            if total_requests > 0:
                for od_key, count in self.od_hourly_frequencies.get(hour, {}).items():
                    origin, dest = od_key.split('→')
                    probability = count / total_requests
                    od_pairs.append({
                        'origin': origin,
                        'destination': dest,
                        'count': count,
                        'probability': probability
                    })

                # Sort by probability (descending)
                od_pairs.sort(key=lambda x: x['probability'], reverse=True)

            self.hourly_stats[hour_key]['od_pairs'] = od_pairs

    def generate_time_range_statistics(self):
        """
        Generate statistics for common time ranges (morning, afternoon, evening, night).

        Returns:
            dict: Time range statistics
        """
        # Define common time ranges
        time_ranges = {
            '08-12': {'name': 'Morning', 'hours': range(8, 12)},
            '12-17': {'name': 'Afternoon', 'hours': range(12, 17)},
            '17-22': {'name': 'Evening', 'hours': range(17, 22)},
            '22-08': {'name': 'Night', 'hours': list(range(22, 24)) + list(range(0, 8))}
        }

        time_range_stats = {}

        for range_key, range_info in time_ranges.items():
            hours = range_info['hours']

            # Collect all requests in this time range
            range_requests = 0
            od_counts = defaultdict(int)
            type_counts = defaultdict(int)
            urgency_counts = defaultdict(int)

            for hour in hours:
                hour_key = f"{hour:02d}-{(hour + 1) % 24:02d}"
                hour_stats = self.hourly_stats.get(hour_key, {})

                # Add request count
                range_requests += hour_stats.get('request_count', 0)

                # Add OD pair counts
                for od_pair in hour_stats.get('od_pairs', []):
                    od_key = f"{od_pair['origin']}→{od_pair['destination']}"
                    od_counts[od_key] += od_pair['count']

                # Add transport type counts
                for t_type, count in hour_stats.get('transport_types', {}).items():
                    type_counts[t_type] += count

                # Add urgency counts
                for urgent, count in hour_stats.get('urgency_distribution', {}).items():
                    urgency_counts[urgent] += count

            # Calculate range statistics
            hours_in_range = len(list(hours))
            requests_per_hour = range_requests / max(1, hours_in_range)

            # Calculate OD pair probabilities
            od_pairs = []
            if range_requests > 0:
                for od_key, count in od_counts.items():
                    origin, dest = od_key.split('→')
                    probability = count / range_requests
                    od_pairs.append({
                        'origin': origin,
                        'destination': dest,
                        'count': count,
                        'probability': probability
                    })

                # Sort by probability (descending)
                od_pairs.sort(key=lambda x: x['probability'], reverse=True)

            # Calculate type and urgency distributions
            type_distribution = {}
            if sum(type_counts.values()) > 0:
                for t_type, count in type_counts.items():
                    type_distribution[t_type] = count / sum(type_counts.values())

            urgency_distribution = {}
            if sum(urgency_counts.values()) > 0:
                for urgent, count in urgency_counts.items():
                    urgency_distribution[urgent] = count / sum(urgency_counts.values())

            # Store range statistics
            time_range_stats[range_key] = {
                'name': range_info['name'],
                'request_count': range_requests,
                'requests_per_hour': requests_per_hour,
                'hours': hours_in_range,
                'od_pairs': od_pairs,
                'transport_type_distribution': type_distribution,
                'urgency_distribution': urgency_distribution
            }

        return time_range_stats

    def export_analysis(self):
        """
        Export analysis results to JSON files.

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        try:
            # 1. Export hourly request statistics
            hourly_request_stats_file = os.path.join(self.output_dir, 'hourly_request_stats.json')
            with open(hourly_request_stats_file, 'w') as f:
                json.dump(self.hourly_stats, f, indent=2)

            # 2. Export time range statistics
            time_range_stats = self.generate_time_range_statistics()
            time_range_stats_file = os.path.join(self.output_dir, 'time_range_stats.json')
            with open(time_range_stats_file, 'w') as f:
                json.dump(time_range_stats, f, indent=2)

            # 3. Export origin-destination pairs by time
            od_pairs_by_time = {}
            for hour_key, stats in self.hourly_stats.items():
                od_pairs_by_time[hour_key] = stats.get('od_pairs', [])

            od_pairs_file = os.path.join(self.output_dir, 'od_pairs_by_time.json')
            with open(od_pairs_file, 'w') as f:
                json.dump(od_pairs_by_time, f, indent=2)

            # 4. Export normalized department names mapping
            name_mapping_file = os.path.join(self.output_dir, 'name_mapping.json')
            with open(name_mapping_file, 'w') as f:
                json.dump(self.name_mapping, f, indent=2)

            # 5. Export metadata
            metadata = {
                'file_analyzed': self.file_path,
                'analysis_date': datetime.now().isoformat(),
                'record_count': len(self.cleaned_data) if hasattr(self, 'cleaned_data') else 0,
                'unique_departments': len(set(self.name_mapping.values())),
                'column_mapping': self.column_mapping
            }

            metadata_file = os.path.join(self.output_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Analysis results exported to {self.output_dir}")
            self.logger.info(f"  - Hourly statistics: {hourly_request_stats_file}")
            self.logger.info(f"  - Time range statistics: {time_range_stats_file}")
            self.logger.info(f"  - OD pairs by time: {od_pairs_file}")

            return True

        except Exception as e:
            self.logger.error(f"Error exporting analysis: {str(e)}")
            return False

    def get_normalized_department_list(self):
        """Get list of normalized department names."""
        return list(set(self.name_mapping.values()))

    def get_hourly_stats(self, hour=None):
        """
        Get hourly statistics.

        Args:
            hour (int, optional): Specific hour (0-23). If None, returns all hours.

        Returns:
            dict: Hourly statistics
        """
        if hour is not None:
            hour_key = f"{hour:02d}-{(hour + 1) % 24:02d}"
            return self.hourly_stats.get(hour_key, {})
        return self.hourly_stats

    def get_time_range_stats(self):
        """Get time range statistics."""
        return self.generate_time_range_statistics()

    def analyze_and_export(self):
        """
        Complete workflow: load, analyze, and export data.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.load_and_prepare_data():
            return False

        if not self.analyze_hourly_patterns():
            return False

        return self.export_analysis()