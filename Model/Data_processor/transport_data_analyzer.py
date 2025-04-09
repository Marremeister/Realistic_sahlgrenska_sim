# Model/data_processor/transport_data_analyzer.py
"""
Class for analyzing hospital transport data from Excel files.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging


class TransportDataAnalyzer:
    """
    Analyzes transport data from Excel files to extract patterns, clean data,
    and prepare for graph building.
    """

    def __init__(self, file_path):
        """
        Initialize the TransportDataAnalyzer with a file path.

        Args:
            file_path (str): Path to the Excel file containing transport data
        """
        self.file_path = file_path
        self.data = None
        self.cleaned_data = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up a logger for the TransportDataAnalyzer."""
        logger = logging.getLogger("TransportDataAnalyzer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_data(self):
        """
        Load data from the file, using semicolon delimiter for CSV.
        """
        self.logger.info(f"Loading data from {self.file_path}")
        try:
            # Check file extension
            if self.file_path.lower().endswith('.csv'):
                # Use semicolon as delimiter with newer pandas parameter name
                self.data = pd.read_csv(
                    self.file_path,
                    delimiter=';',
                    on_bad_lines='warn',  # For newer pandas versions
                    low_memory=False  # Better for inconsistent data
                )
            else:
                # Excel file
                self.data = pd.read_excel(self.file_path)

            self.logger.info(f"Loaded {len(self.data)} rows of data with {len(self.data.columns)} columns")
            self.logger.info(f"Columns: {list(self.data.columns)}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def clean_data(self):
        """
        Clean the data by removing invalid entries, such as:
        - Negative transport times
        - Missing origin/destination
        - Extremely long transport times (outliers)

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.data is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return None

        self.logger.info("Cleaning data...")
        df = self.data.copy()

        # Record initial size
        initial_size = len(df)

        # Map Swedish column names to expected ones
        # Based on the columns we saw in the logs
        column_mapping = {
            'origin': 'Startplats',  # Origin location
            'destination': 'Slutplats',  # Destination location
            'start': 'Starttid',  # Start time
            'end': 'Uppdrag Sluttid'  # End time
        }

        # Log the identified columns
        self.logger.info(f"Using columns: Origin={column_mapping['origin']}, "
                         f"Destination={column_mapping['destination']}, "
                         f"Start Time={column_mapping['start']}, "
                         f"End Time={column_mapping['end']}")

        # Remove rows with missing origin or destination
        if column_mapping['origin'] in df.columns and column_mapping['destination'] in df.columns:
            df.dropna(subset=[column_mapping['origin'], column_mapping['destination']], inplace=True)
            self.logger.info(f"Removed {initial_size - len(df)} rows with missing origin/destination")
            initial_size = len(df)
        else:
            self.logger.warning(f"Could not find expected origin/destination columns")
            self.logger.info(f"Available columns: {', '.join(df.columns[:10])}... (and more)")

        # Calculate transport times if start and end times exist
        if column_mapping['start'] in df.columns and column_mapping['end'] in df.columns:
            # Try to convert to datetime with explicit European format
            for col in [column_mapping['start'], column_mapping['end']]:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        # Use explicit format: DD-MM-YYYY HH:MM:SS
                        df[col] = pd.to_datetime(df[col], format='%d-%m-%Y %H:%M:%S')
                    except Exception as e:
                        self.logger.warning(f"Could not convert {col} to datetime: {str(e)}")

                        # Try with dayfirst=True as a fallback
                        try:
                            df[col] = pd.to_datetime(df[col], dayfirst=True)
                        except Exception as e2:
                            self.logger.error(f"Fallback datetime conversion also failed: {str(e2)}")

            # Calculate transport time in seconds
            try:
                df['transport_time'] = (df[column_mapping['end']] - df[column_mapping['start']]).dt.total_seconds()

                # Remove negative transport times
                negative_count = len(df[df['transport_time'] < 0])
                df = df[df['transport_time'] >= 0]
                self.logger.info(f"Removed {negative_count} rows with negative transport times")

                # Remove extreme outliers (transport times more than 3 hours)
                outlier_count = len(df[df['transport_time'] > 3 * 60 * 60])
                df = df[df['transport_time'] <= 3 * 60 * 60]
                self.logger.info(f"Removed {outlier_count} rows with transport times > 3 hours")
            except Exception as e:
                self.logger.error(f"Error calculating transport times: {str(e)}")
                # If transport_time calculation fails, create a default one
                try:
                    # Add a placeholder transport_time column
                    df['transport_time'] = 300  # Default 5 minutes
                    self.logger.warning("Created placeholder transport_time column with default values")
                except Exception as e2:
                    self.logger.error(f"Could not create placeholder transport_time: {str(e2)}")
        else:
            self.logger.warning(f"Could not find expected start/end time columns")

        self.cleaned_data = df
        self.logger.info(f"Data cleaning complete. Remaining rows: {len(df)}")

        # Store column mapping for other methods to use
        self._column_mapping = column_mapping

        return df

    def _find_column_containing(self, keyword, columns):
        """Find a column matching a given keyword or use predefined mapping."""
        # First check if we have a mapping
        if hasattr(self, '_column_mapping') and keyword in self._column_mapping:
            mapped_col = self._column_mapping[keyword]
            if mapped_col in columns:
                return mapped_col

        # Fall back to the original method
        matches = [col for col in columns if keyword.lower() in col.lower()]
        return matches[0] if matches else None

    def get_origin_destination_pairs(self):
        """
        Get all unique origin-destination pairs from the cleaned data.

        Returns:
            list: List of (origin, destination) tuples
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return []

        # Use the mapping from clean_data if available
        if hasattr(self, '_column_mapping'):
            origin_col = self._column_mapping.get('origin')
            dest_col = self._column_mapping.get('destination')
        else:
            origin_col = self._find_column_containing('origin', self.cleaned_data.columns)
            dest_col = self._find_column_containing('destination', self.cleaned_data.columns)

        if not origin_col or not dest_col:
            self.logger.error("Could not identify origin or destination columns.")
            return []

        if origin_col not in self.cleaned_data.columns or dest_col not in self.cleaned_data.columns:
            self.logger.error(f"Columns {origin_col} or {dest_col} not found in DataFrame.")
            self.logger.info(f"Available columns: {', '.join(self.cleaned_data.columns[:10])}... (and more)")
            return []

        pairs = self.cleaned_data[[origin_col, dest_col]].drop_duplicates()
        return list(zip(pairs[origin_col], pairs[dest_col]))

    def get_median_transport_times(self):
        """
        Calculate median transport times for each origin-destination pair.

        Returns:
            dict: Dictionary mapping (origin, destination) tuples to median transport times in seconds
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return {}

        origin_col = self._find_column_containing('origin', self.cleaned_data.columns)
        dest_col = self._find_column_containing('destination', self.cleaned_data.columns)

        if not origin_col or not dest_col:
            self.logger.error("Could not identify origin or destination columns.")
            return {}

        # Check if transport_time column exists
        if 'transport_time' not in self.cleaned_data.columns:
            self.logger.warning("Transport_time column not found. Generating placeholder transport times.")
            # Create a placeholder with fixed times
            result = {}
            for origin, dest in self.get_origin_destination_pairs():
                result[(origin, dest)] = 300  # 5 minutes in seconds
            return result

        # Group by origin-destination and calculate median
        try:
            grouped = self.cleaned_data.groupby([origin_col, dest_col])['transport_time'].median()
            return grouped.to_dict()
        except Exception as e:
            self.logger.error(f"Error calculating median transport times: {str(e)}")
            # Fallback to placeholder values
            result = {}
            for origin, dest in self.get_origin_destination_pairs():
                result[(origin, dest)] = 300  # 5 minutes in seconds
            return result

    def get_fastest_times(self, percentile=10):
        """
        Get the fastest times for each origin-destination pair (based on a percentile).
        This helps identify the most realistic travel times by focusing on the fastest transports.

        Args:
            percentile (int): Percentile to use (e.g., 10 for the 10% fastest transports)

        Returns:
            dict: Dictionary mapping (origin, destination) tuples to fast transport times
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return {}

        origin_col = self._find_column_containing('origin', self.cleaned_data.columns)
        dest_col = self._find_column_containing('destination', self.cleaned_data.columns)

        if not origin_col or not dest_col:
            self.logger.error("Could not identify origin or destination columns.")
            return {}

        # Group by origin-destination and calculate the specified percentile
        # Using np.percentile instead of pandas quantile for more flexibility
        result = {}
        for (origin, dest), group in self.cleaned_data.groupby([origin_col, dest_col]):
            transport_times = group['transport_time'].values
            if len(transport_times) > 3:  # Only consider pairs with sufficient data
                fast_time = np.percentile(transport_times, percentile)
                result[(origin, dest)] = fast_time

        return result

    def get_hourly_request_distribution(self):
        """
        Get the distribution of requests by hour of the day.

        Returns:
            dict: Dictionary mapping hour (0-23) to number of requests
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return {}

        start_time_col = self._find_column_containing('start', self.cleaned_data.columns)
        if not start_time_col:
            self.logger.error("Could not identify start time column.")
            return {}

        try:
            # Check if the column is already in datetime format
            if pd.api.types.is_datetime64_any_dtype(self.cleaned_data[start_time_col]):
                # Extract hour from start time
                hours = self.cleaned_data[start_time_col].dt.hour
                distribution = hours.value_counts().sort_index().to_dict()
                return distribution
            else:
                # Try to convert to datetime first
                try:
                    datetime_col = pd.to_datetime(self.cleaned_data[start_time_col], format='%d-%m-%Y %H:%M:%S')
                    hours = datetime_col.dt.hour
                    distribution = hours.value_counts().sort_index().to_dict()
                    return distribution
                except Exception as e:
                    # If that fails, extract hour manually from string
                    self.logger.warning(f"Could not convert to datetime for hourly distribution: {str(e)}")

                    # Try to extract hour from string format "DD-MM-YYYY HH:MM:SS"
                    try:
                        # Use string extraction
                        hour_strings = self.cleaned_data[start_time_col].str.extract(r'\d+-\d+-\d+ (\d+):\d+:\d+')[0]
                        hours = pd.to_numeric(hour_strings, errors='coerce')
                        distribution = hours.value_counts().sort_index().to_dict()
                        return distribution
                    except Exception as e2:
                        self.logger.error(f"Manual hour extraction failed: {str(e2)}")

                        # Return a default distribution if all else fails
                        return {h: 100 for h in range(24)}
        except Exception as e:
            self.logger.error(f"Error calculating hourly distribution: {str(e)}")
            return {h: 100 for h in range(24)}  # Default fallback

    def get_all_departments(self):
        """
        Get all unique departments (nodes) in the data.

        Returns:
            list: List of department names
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return []

        origin_col = self._find_column_containing('origin', self.cleaned_data.columns)
        dest_col = self._find_column_containing('destination', self.cleaned_data.columns)

        if not origin_col or not dest_col:
            self.logger.error("Could not identify origin or destination columns.")
            return []

        origins = set(self.cleaned_data[origin_col].unique())
        destinations = set(self.cleaned_data[dest_col].unique())

        return list(origins.union(destinations))

    def get_frequency_matrix(self):
        """
        Create a frequency matrix showing how often each origin-destination pair occurs.

        Returns:
            pd.DataFrame: Matrix of transport frequencies
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return None

        origin_col = self._find_column_containing('origin', self.cleaned_data.columns)
        dest_col = self._find_column_containing('destination', self.cleaned_data.columns)

        if not origin_col or not dest_col:
            self.logger.error("Could not identify origin or destination columns.")
            return None

        # Count occurrences of each origin-destination pair
        frequency = self.cleaned_data.groupby([origin_col, dest_col]).size().reset_index()
        frequency.columns = [origin_col, dest_col, 'frequency']

        # Convert to a pivot table for better visualization
        pivot = frequency.pivot(index=origin_col, columns=dest_col, values='frequency')

        # Fill NaN with 0
        pivot = pivot.fillna(0)

        return pivot

    def get_time_range_distribution(self, start_hour, end_hour):
        """
        Get the distribution of requests within a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            dict: Statistics about requests in the given time range
        """
        if self.cleaned_data is None:
            self.logger.error("No cleaned data available. Call clean_data() first.")
            return {}

        start_time_col = self._find_column_containing('start', self.cleaned_data.columns)
        if not start_time_col:
            self.logger.error("Could not identify start time column.")
            return {}

        try:
            # Filter data for the given time range
            filtered_data = self._filter_by_time_range(start_hour, end_hour, start_time_col)
            if filtered_data.empty:
                self.logger.warning(f"No data found for time range {start_hour}-{end_hour}.")
                return {}

            # Calculate basic statistics
            total_requests = len(filtered_data)

            # Get origin-destination distribution in this time range
            origin_col = self._find_column_containing('origin', filtered_data.columns)
            dest_col = self._find_column_containing('destination', filtered_data.columns)

            if not origin_col or not dest_col:
                self.logger.error("Could not identify origin or destination columns.")
                return {}

            # Count occurrences of each origin-destination pair
            od_counts = filtered_data.groupby([origin_col, dest_col]).size()
            od_distribution = (od_counts / total_requests).to_dict()

            # Calculate transport type and urgency distributions if available
            type_distribution = {}
            urgency_distribution = {}

            type_col = self._find_column_containing('type', filtered_data.columns)
            if type_col:
                type_counts = filtered_data[type_col].value_counts()
                type_distribution = (type_counts / total_requests).to_dict()

            urgency_col = self._find_column_containing('urgent', filtered_data.columns)
            if urgency_col:
                urgency_counts = filtered_data[urgency_col].value_counts()
                urgency_distribution = (urgency_counts / total_requests).to_dict()

            # Calculate hourly rate within the range
            hours_in_range = end_hour - start_hour
            if hours_in_range <= 0:  # Handle wrap-around (e.g., 22-6)
                hours_in_range += 24
            hourly_rate = total_requests / hours_in_range

            return {
                'total_requests': total_requests,
                'hourly_rate': hourly_rate,
                'od_distribution': od_distribution,
                'type_distribution': type_distribution,
                'urgency_distribution': urgency_distribution
            }
        except Exception as e:
            self.logger.error(f"Error analyzing time range distribution: {str(e)}")
            return {}

    def _filter_by_time_range(self, start_hour, end_hour, time_column):
        """
        Filter data by time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            time_column (str): Column name containing time data

        Returns:
            pd.DataFrame: Filtered data
        """
        # Check if the column is already in datetime format
        if pd.api.types.is_datetime64_any_dtype(self.cleaned_data[time_column]):
            # Extract hour from time
            hour_series = self.cleaned_data[time_column].dt.hour

            if start_hour <= end_hour:
                # Simple case: e.g., 8-15
                mask = (hour_series >= start_hour) & (hour_series < end_hour)
            else:
                # Wrap-around case: e.g., 22-6
                mask = (hour_series >= start_hour) | (hour_series < end_hour)

            return self.cleaned_data[mask]
        else:
            # Try to convert to datetime first
            try:
                datetime_col = pd.to_datetime(self.cleaned_data[time_column], format='%d-%m-%Y %H:%M:%S')
                hour_series = datetime_col.dt.hour

                if start_hour <= end_hour:
                    mask = (hour_series >= start_hour) & (hour_series < end_hour)
                else:
                    mask = (hour_series >= start_hour) | (hour_series < end_hour)

                return self.cleaned_data[mask]
            except Exception as e:
                self.logger.warning(f"Could not convert to datetime for filtering: {str(e)}")
                return pd.DataFrame()  # Return empty DataFrame if conversion fails

    def get_od_pair_probabilities_by_time(self, start_hour, end_hour):
        """
        Get probability distribution of origin-destination pairs for a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            list: List of (origin, destination, probability) tuples, sorted by probability
        """
        stats = self.get_time_range_distribution(start_hour, end_hour)
        if not stats or 'od_distribution' not in stats:
            return []

        od_dist = stats['od_distribution']
        result = [(*pair, prob) for pair, prob in od_dist.items()]
        return sorted(result, key=lambda x: x[2], reverse=True)

    def get_transport_type_probabilities_by_time(self, start_hour, end_hour):
        """
        Get probability distribution of transport types for a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            dict: Dictionary mapping transport types to probabilities
        """
        stats = self.get_time_range_distribution(start_hour, end_hour)
        return stats.get('type_distribution', {})

    def get_urgency_probabilities_by_time(self, start_hour, end_hour):
        """
        Get probability distribution of urgency for a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            dict: Dictionary mapping urgency values to probabilities
        """
        stats = self.get_time_range_distribution(start_hour, end_hour)
        return stats.get('urgency_distribution', {})

    def get_request_rate_by_time_range(self, start_hour, end_hour):
        """
        Get the average request rate (requests per hour) for a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)

        Returns:
            float: Average requests per hour
        """
        stats = self.get_time_range_distribution(start_hour, end_hour)
        return stats.get('hourly_rate', 0)

    def generate_realistic_requests(self, start_hour, end_hour, num_requests=None):
        """
        Generate a list of realistic transport requests for a specific time range.

        Args:
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            num_requests (int, optional): Number of requests to generate.
                                         If None, uses the average rate for the time period.

        Returns:
            list: List of (origin, destination, transport_type, urgent) tuples
        """
        import random

        # Get statistics for the time range
        stats = self.get_time_range_distribution(start_hour, end_hour)
        if not stats or 'od_distribution' not in stats:
            self.logger.error(f"Could not get distribution for time range {start_hour}-{end_hour}.")
            return []

        # Determine number of requests to generate
        if num_requests is None:
            hours_in_range = end_hour - start_hour
            if hours_in_range <= 0:  # Handle wrap-around
                hours_in_range += 24
            num_requests = int(stats.get('hourly_rate', 10) * hours_in_range)

        # Extract distributions
        od_dist = stats.get('od_distribution', {})
        type_dist = stats.get('type_distribution', {})
        urgency_dist = stats.get('urgency_distribution', {})

        # Create OD pairs with their probabilities
        od_pairs = []
        od_probs = []
        for (origin, dest), prob in od_dist.items():
            od_pairs.append((origin, dest))
            od_probs.append(prob)

        # Normalize probabilities if needed
        if od_probs and sum(od_probs) > 0:
            od_probs = [p / sum(od_probs) for p in od_probs]
        else:
            # If no probabilities, use equal distribution
            od_pairs = self.get_origin_destination_pairs()
            od_probs = [1 / len(od_pairs)] * len(od_pairs)

        # Create transport types with probabilities
        # If no type data, use default distribution
        if not type_dist:
            type_dist = {'stretcher': 0.5, 'wheelchair': 0.3, 'bed': 0.2}

        types = list(type_dist.keys())
        type_probs = list(type_dist.values())

        # Normalize type probabilities
        if sum(type_probs) > 0:
            type_probs = [p / sum(type_probs) for p in type_probs]

        # Create urgency probabilities
        # If no urgency data, use default distribution
        if not urgency_dist:
            urgency_dist = {True: 0.2, False: 0.8}

        urgencies = list(urgency_dist.keys())
        urgency_probs = list(urgency_dist.values())

        # Normalize urgency probabilities
        if sum(urgency_probs) > 0:
            urgency_probs = [p / sum(urgency_probs) for p in urgency_probs]

        # Generate requests
        requests = []
        for _ in range(num_requests):
            if not od_pairs:  # Safety check
                break

            # Randomly select O-D pair based on probability distribution
            origin, dest = random.choices(od_pairs, weights=od_probs, k=1)[0]

            # Randomly select transport type
            if types:
                transport_type = random.choices(types, weights=type_probs, k=1)[0]
            else:
                transport_type = 'stretcher'  # Default

            # Randomly select urgency
            if urgencies:
                urgent = random.choices(urgencies, weights=urgency_probs, k=1)[0]
            else:
                urgent = False  # Default

            requests.append((origin, dest, transport_type, urgent))

        return requests

    def export_time_based_statistics(self, output_dir='analysis_output'):
        """
        Export time-based statistics for all hourly ranges to files.

        Args:
            output_dir (str): Directory to save the statistics

        Returns:
            bool: True if successful
        """
        import os
        import json

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Store hourly statistics
            hourly_stats = {}

            # Single hour ranges
            for hour in range(24):
                stats = self.get_time_range_distribution(hour, (hour + 1) % 24)
                if stats:
                    hourly_stats[f"{hour:02d}-{(hour + 1) % 24:02d}"] = {
                        'requests_per_hour': stats.get('hourly_rate', 0),
                        'total_requests': stats.get('total_requests', 0)
                    }

            # Common time ranges
            time_ranges = [
                (8, 12),  # Morning
                (12, 17),  # Afternoon
                (17, 22),  # Evening
                (22, 8)  # Night
            ]

            range_stats = {}
            for start, end in time_ranges:
                stats = self.get_time_range_distribution(start, end)
                if stats:
                    range_stats[f"{start:02d}-{end:02d}"] = {
                        'requests_per_hour': stats.get('hourly_rate', 0),
                        'total_requests': stats.get('total_requests', 0),
                        'hours': end - start if end > start else end + 24 - start
                    }

            # Save hourly statistics
            with open(os.path.join(output_dir, 'hourly_request_stats.json'), 'w') as f:
                json.dump(hourly_stats, f, indent=2)

            # Save time range statistics
            with open(os.path.join(output_dir, 'time_range_stats.json'), 'w') as f:
                json.dump(range_stats, f, indent=2)

            # Export top origin-destination pairs for each time range
            od_by_time = {}
            for start, end in time_ranges:
                top_pairs = self.get_od_pair_probabilities_by_time(start, end)
                # Take top 20 pairs or all if less than 20
                top_n = min(20, len(top_pairs))
                od_by_time[f"{start:02d}-{end:02d}"] = [
                    {
                        'origin': origin,
                        'destination': dest,
                        'probability': prob
                    }
                    for origin, dest, prob in top_pairs[:top_n]
                ]

            # Save OD pair probabilities by time range
            with open(os.path.join(output_dir, 'od_pairs_by_time.json'), 'w') as f:
                json.dump(od_by_time, f, indent=2)

            self.logger.info(f"Time-based statistics exported to {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting time-based statistics: {str(e)}")
            return False