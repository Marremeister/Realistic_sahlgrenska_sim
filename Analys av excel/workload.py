import pandas as pd
import time
from plotters import Plot # Assuming plotters.py is correct

class HourlyWorkloadAnalysis:
    def __init__(self, processed_df, plotter=None): # Expects processed_df now
        """Initializes with the processed DataFrame."""
        if processed_df is None or processed_df.empty:
             raise ValueError("Input DataFrame for HourlyWorkloadAnalysis is empty or None.")
        # Essential columns for workload analysis:
        required_cols = ['Real Start Time', 'Uppdrag Sluttid', 'Sekundär Servicepersonal Id', 'Skapad Tid']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
             raise ValueError(f"DataFrame missing required columns for Workload Analysis: {missing_cols}")

        self.df = processed_df # Store the processed DataFrame
        self.plotter = plotter
        self.hourly_data = None
        self.overall_transporter_workload = None

    def _expand_hours(self, df_segment):
        """
        Expand tasks into hourly intervals based on 'Real Start Time' and 'Uppdrag Sluttid'.
        Calculates 'Effective_Time_Per_Hour'.
        (Internal logic remains largely the same, operates on the input df_segment)
        """
        df = df_segment.copy() # Work on a copy of the segment
        start_time = time.perf_counter()
        print(f"  Expanding hours for segment with {len(df)} rows...")

        # Check if required columns exist in this segment
        if not all(col in df.columns for col in ['Real Start Time', 'Uppdrag Sluttid']):
            print("  Error: Missing 'Real Start Time' or 'Uppdrag Sluttid' in segment for expansion.")
            return pd.DataFrame() # Return empty DataFrame

        # --- Step 1: Compute start_hour and adjusted_end ---
        df['start_hour'] = df['Real Start Time'].dt.floor('h')
        # Adjust end time if it falls exactly on the hour mark to avoid double counting
        df['adjusted_end'] = df['Uppdrag Sluttid']
        mask = (df['Uppdrag Sluttid'].dt.minute == 0) & \
               (df['Uppdrag Sluttid'].dt.second == 0) & \
               (df['Uppdrag Sluttid'].dt.microsecond == 0)
        df.loc[mask, 'adjusted_end'] = df.loc[mask, 'Uppdrag Sluttid'] - pd.Timedelta(microseconds=1)
        df['end_hour'] = df['adjusted_end'].dt.floor('h')

        # --- Step 2: Generate hour_range (Handle potential NaTs) ---
        df.dropna(subset=['start_hour', 'end_hour'], inplace=True) # Drop rows where floor('h') failed
        if df.empty:
            print("  Segment empty after dropping NaT start/end hours.")
            return pd.DataFrame()

        # Handle cases where start_hour > end_hour (data issue?)
        df = df[df['start_hour'] <= df['end_hour']]
        if df.empty:
             print("  Segment empty after filtering start_hour <= end_hour.")
             return pd.DataFrame()

        # Apply pd.date_range
        hour_ranges = []
        for _, row in df.iterrows():
             try:
                 # Explicitly handle potential NaT again just in case
                 if pd.isna(row['start_hour']) or pd.isna(row['end_hour']):
                     hour_ranges.append([])
                 else:
                      # Use closed='left' if available/needed, or handle boundary logic carefully
                      # freq='h' or 'H'
                      hour_ranges.append(pd.date_range(row['start_hour'], row['end_hour'], freq='H'))
             except Exception as e:
                  print(f"  Warning: Error generating date range for row: {e}. Row index: {_}")
                  hour_ranges.append([]) # Append empty list on error
        df['hour_range'] = hour_ranges


        # --- Step 3: Explode ---
        # Filter out rows with empty hour_range before exploding
        df = df[df['hour_range'].apply(lambda x: len(x) > 0)]
        if df.empty:
             print("  Segment empty after filtering empty hour ranges.")
             return pd.DataFrame()
        df_expanded = df.explode('hour_range')

        # --- Step 4: Active_Hour, Active_Date ---
        df_expanded['Active_Hour'] = df_expanded['hour_range'].dt.hour
        df_expanded['Active_Date'] = df_expanded['hour_range'].dt.date # Keep date for potential daily aggregation

        # --- Step 5: next_hour ---
        df_expanded['next_hour'] = df_expanded['hour_range'] + pd.Timedelta(hours=1)

        # --- Step 6: interval_start, interval_end ---
        df_expanded['interval_start'] = df_expanded[['Real Start Time', 'hour_range']].max(axis=1)
        df_expanded['interval_end'] = df_expanded[['adjusted_end', 'next_hour']].min(axis=1)

        # --- Step 7: Effective_Time_Per_Hour ---
        # Ensure interval_end > interval_start before calculating duration
        df_expanded = df_expanded[df_expanded['interval_end'] > df_expanded['interval_start']]
        if df_expanded.empty:
            print("  Segment empty after interval filtering.")
            return pd.DataFrame()

        df_expanded['Effective_Time_Per_Hour'] = (
                (df_expanded['interval_end'] - df_expanded['interval_start']).dt.total_seconds() / 60
        )
        # Filter out tiny negative values due to float precision if necessary
        df_expanded = df_expanded[df_expanded['Effective_Time_Per_Hour'] >= 0]


        total_time = time.perf_counter() - start_time
        print(f"  Expansion completed in {total_time:.2f} seconds. Result rows: {len(df_expanded)}")
        return df_expanded

    def calculate_workload_data(self):
        """
        Calculate workload data for selected hours using the preprocessed DataFrame.
        Filters for January based on 'Skapad Tid'.
        """
        print("Calculating workload data...")
        # Use the processed DataFrame stored in self.df
        january_df = self.df[self.df['Skapad Tid'].dt.month == 1]
        print(f"Filtered for January: {len(january_df)} rows (out of {len(self.df)} total).")
        if january_df.empty:
            print("No data found for January. Cannot calculate workload.")
            self.hourly_data = {}
            self.overall_transporter_workload = pd.Series(dtype=float) # Empty series
            return {}, self.overall_transporter_workload

        selected_hours = [2, 8, 12, 14, 16, 20]
        hourly_data = {}
        expanded_list = []

        # Iterate through hours and expand relevant data segments
        for hour in selected_hours:
            print(f"\nProcessing hour: {hour}")
            # Select tasks that *overlap* with the hour
            # Task starts before/during the hour AND ends during/after the hour
            # Note: Using Real Start Time here as per the logic of workload calculation
            hour_start_dt = pd.Timestamp.min.replace(hour=hour) # Represents start of hour boundary
            hour_end_dt = pd.Timestamp.min.replace(hour=hour+1 if hour < 23 else hour) # Represents end

            # Find tasks potentially overlapping the hour
            # A task overlaps if: start_time < hour_end AND end_time > hour_start
            # We use floor/ceil logic in _expand_hours, but a broader filter here is safer.
            # Consider tasks starting up to an hour before, and ending any time after the hour starts.
            potential_overlap_df = january_df[
                 (january_df['Real Start Time'].dt.hour <= hour) &
                 (january_df['Uppdrag Sluttid'].dt.hour >= hour)
            ]

            # More precise filter:
            # Select tasks where [Real Start Time, Uppdrag Sluttid] overlaps with [hour:00, hour+1:00)
            # Condition: RealStartTime < HourEndTime AND UppdragSluttid > HourStartTime
            # We approximate HourStartTime/HourEndTime conceptually
            trial_df = january_df[
                (january_df['Real Start Time'].dt.hour <= hour) & # Starts before or during the hour
                (january_df['Uppdrag Sluttid'] >= january_df['Real Start Time'].dt.floor('h').apply(lambda dt: dt.replace(hour=hour))) # Ends at or after the hour starts
                 # Ensure start time exists
            ].dropna(subset=['Real Start Time'])

            print(f"Initial rows overlapping hour {hour}: {len(trial_df)}")
            if trial_df.empty:
                print(f"No tasks overlap hour {hour}.")
                continue

            # Expand this segment
            expanded_df_for_hour = self._expand_hours(trial_df)

            if expanded_df_for_hour.empty:
                 print(f"Expansion yielded no data for hour {hour}.")
                 continue

            # Filter the *expanded* data to keep only rows where Active_Hour is the target hour
            hour_specific_expanded_df = expanded_df_for_hour[expanded_df_for_hour['Active_Hour'] == hour]

            if hour_specific_expanded_df.empty:
                 print(f"No expanded rows correspond exactly to Active_Hour {hour}.")
                 continue

            print(f"Data specific to Active_Hour {hour}: {len(hour_specific_expanded_df)} rows.")
            hourly_data[hour] = hour_specific_expanded_df
            expanded_list.append(hour_specific_expanded_df)

        # Calculate overall average workload based on the combined expanded data for selected hours
        if expanded_list:
            combined_df = pd.concat(expanded_list)
            if 'Sekundär Servicepersonal Id' in combined_df.columns and 'Effective_Time_Per_Hour' in combined_df.columns:
                grouped = combined_df.groupby('Sekundär Servicepersonal Id')
                total_workload = grouped['Effective_Time_Per_Hour'].sum()
                count_instances = grouped.size() # Counts rows per transporter in the expanded data
                # Avoid division by zero if count_instances is 0 for a transporter (shouldn't happen with groupby)
                overall_avg_workload = (total_workload / count_instances.replace(0, 1)).fillna(0)
            else:
                print("Warning: Missing 'Sekundär Servicepersonal Id' or 'Effective_Time_Per_Hour' in combined expanded data.")
                overall_avg_workload = pd.Series(dtype=float)
        else:
            print("No data was successfully expanded for any selected hour.")
            overall_avg_workload = pd.Series(dtype=float)

        self.hourly_data = hourly_data
        self.overall_transporter_workload = overall_avg_workload
        print("Workload data calculation finished.")
        # Return values are not strictly needed if stored in self, but can be useful
        return self.hourly_data, self.overall_transporter_workload

    def plot_hourly_workload(self, plotter=None):
        """
        Plot histograms for each selected hour using pre-calculated hourly_data.
        """
        plotter = plotter if plotter is not None else self.plotter
        if plotter is None:
             print("Error: No plotter provided or set for plot_hourly_workload.")
             return
        if self.hourly_data is None:
             print("Workload data not calculated yet. Call calculate_workload_data() first.")
             # Optionally, call it here: self.calculate_workload_data()
             return

        print("Plotting hourly workload histograms...")
        if not self.hourly_data:
            print("No hourly data available to plot.")
            return

        for hour, df_hour in self.hourly_data.items():
            if df_hour.empty or 'Sekundär Servicepersonal Id' not in df_hour.columns or 'Effective_Time_Per_Hour' not in df_hour.columns:
                 print(f"Skipping plot for hour {hour}: Data missing or empty.")
                 continue

            grouped = df_hour.groupby('Sekundär Servicepersonal Id')
            total_workload = grouped['Effective_Time_Per_Hour'].sum()
            instance_count = grouped.size()
            # Avoid division by zero
            average_workload = (total_workload / instance_count.replace(0, 1)).fillna(0)

            if average_workload.empty:
                 print(f"Skipping plot for hour {hour}: No average workload data.")
                 continue

            median_workload = average_workload.median()
            title = f"Workload at {hour}:00 (Jan)\nMedian: {median_workload:.1f} min per instance"
            plot = Plot(
                xlabel='Average Workload (min per instance)',
                ylabel='Number of Transporters',
                title=title,
                plot_type='hist',
                x=average_workload,
                bins=20, # Or dynamic bins based on data range
                median=median_workload
            )
            plotter.add_plot(plot)
        print("Finished adding hourly workload plots.")


    def plot_transporter_workload_by_hour(self, plotter=None):
        """
        Plot bar charts for each selected hour using pre-calculated hourly_data.
        """
        plotter = plotter if plotter is not None else self.plotter
        if plotter is None:
             print("Error: No plotter provided or set for plot_transporter_workload_by_hour.")
             return
        if self.hourly_data is None:
             print("Workload data not calculated yet. Call calculate_workload_data() first.")
             return

        print("Plotting transporter workload bar charts...")
        if not self.hourly_data:
            print("No hourly data available to plot.")
            return

        for hour, df_hour in self.hourly_data.items():
            if df_hour.empty or 'Sekundär Servicepersonal Id' not in df_hour.columns or 'Effective_Time_Per_Hour' not in df_hour.columns:
                 print(f"Skipping plot for hour {hour}: Data missing or empty.")
                 continue

            grouped = df_hour.groupby('Sekundär Servicepersonal Id')
            total_workload = grouped['Effective_Time_Per_Hour'].sum()
            instance_count = grouped.size()
            # Avoid division by zero
            average_workload = (total_workload / instance_count.replace(0, 1)).fillna(0)

            if average_workload.empty:
                 print(f"Skipping plot for hour {hour}: No average workload data.")
                 continue

            # Sort for better visualization if many transporters
            average_workload_sorted = average_workload.sort_values(ascending=False)

            title = f"Transporter Workload at {hour}:00 (January)"
            plot = Plot(
                xlabel='Transporter ID',
                ylabel='Average Workload (min per instance)',
                title=title,
                plot_type='bar',
                # Plot sorted data
                x=average_workload_sorted.index.astype(str), # Ensure x-axis labels are strings
                y=average_workload_sorted.values
            )
            plotter.add_plot(plot)
        print("Finished adding transporter workload plots.")