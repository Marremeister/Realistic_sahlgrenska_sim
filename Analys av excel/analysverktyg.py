import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time # Added for timing preprocessing
import seaborn as sns # Added for the demand plot

# ======================================
#        Data Loading Class
# ======================================
class DataLoader:
    """Loads raw data and performs essential datetime conversions."""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Loads the CSV file and converts essential time columns to datetime.
        Handles basic loading errors and drops rows missing essential times.

        Returns:
            pd.DataFrame or None: The loaded DataFrame with basic time conversions,
                                 or None if loading fails.
        """
        try:
            df = pd.read_csv(
                self.file_path,
                sep=";",
                on_bad_lines='skip',
                encoding="utf-8",
                low_memory=False
            )
            print(f"Successfully loaded {len(df)} rows from {self.file_path}")
            print("Raw columns found:", df.columns.tolist())

            # --- Essential Time Columns ---
            time_cols_to_convert = ['Skapad Tid', 'Uppdrag Sluttid', 'Starttid']
            missing_time_cols = [col for col in time_cols_to_convert if col not in df.columns]
            if missing_time_cols:
                print(f"Warning: Missing essential time columns: {missing_time_cols}. Preprocessing might fail.")

            existing_time_cols = [col for col in time_cols_to_convert if col in df.columns]
            for col in existing_time_cols:
                print(f"Converting column to datetime: {col}")
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

            # --- Drop rows with missing essential times ---
            initial_rows = len(df)
            df = df.dropna(subset=existing_time_cols)
            if len(df) < initial_rows:
                print(f"Dropped {initial_rows - len(df)} rows due to missing essential datetime values in {existing_time_cols}.")

            if df.empty:
                print("Error: DataFrame is empty after initial loading and datetime handling.")
                return None

            return df

        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            return None
        except Exception as e:
            print(f"Error during data loading: {e}")
            return None

# ======================================
#     Data Preprocessing Class
# ======================================
class DataPreprocessor:
    """Handles complex calculations and filtering on the raw data."""
    def __init__(self, raw_df):
        if raw_df is None or raw_df.empty:
             raise ValueError("Input DataFrame for Preprocessor is empty or None.")
        self.df = raw_df.copy() # Work on a copy

    def preprocess_data(self, route_percentage=10):
        """
        Performs calculations like actual duration, benchmarks, real times,
        and filtering.

        Parameters:
            route_percentage (float): Percentage for benchmark calculation.

        Returns:
            pd.DataFrame or None: The processed DataFrame, or None if errors occur.
        """
        start_proc_time = time.time()
        print("\n--- Starting Data Preprocessing ---")
        try:
            # --- Check Required Columns ---
            required_for_processing = ['Starttid', 'Uppdrag Sluttid', 'Skapad Tid', 'Startplats', 'Slutplats']
            missing_cols = [col for col in required_for_processing if col not in self.df.columns]
            if missing_cols:
                print(f"Error: Preprocessing cannot continue. Missing required columns: {missing_cols}")
                return None

            # --- Calculate Actual Transport Duration ---
            print("Calculating Actual Transport Duration...")
            self.df['Actual Transport Duration (seconds)'] = (self.df['Uppdrag Sluttid'] - self.df['Starttid']).dt.total_seconds()
            self.df['Actual Transport Duration (minutes)'] = self.df['Actual Transport Duration (seconds)'] / 60
            # Handle negative durations (set to 0 or NaN, 0 might be better for workload)
            neg_duration_count = (self.df['Actual Transport Duration (minutes)'] < 0).sum()
            if neg_duration_count > 0:
                 print(f"Warning: Found {neg_duration_count} negative actual durations. Setting them to 0.")
                 self.df.loc[self.df['Actual Transport Duration (minutes)'] < 0, 'Actual Transport Duration (minutes)'] = 0
                 self.df.loc[self.df['Actual Transport Duration (seconds)'] < 0, 'Actual Transport Duration (seconds)'] = 0


            # --- Calculate Route Benchmark Duration ---
            print(f"Calculating route benchmark (median of fastest {route_percentage}% durations)...")
            # Filter out non-positive durations before calculating benchmark
            positive_duration_df = self.df[self.df['Actual Transport Duration (seconds)'] > 0].copy()

            if not positive_duration_df.empty:
                 # Calculate the median duration (in seconds) of the fastest tasks per route
                 route_time_benchmarks_seconds = positive_duration_df.groupby(['Startplats', 'Slutplats'])['Actual Transport Duration (seconds)'].apply(
                     lambda x: x.nsmallest(max(1, int(len(x) * (route_percentage / 100)))).median() if not x.empty else np.nan
                 )
                 # Convert benchmark to minutes for the column name convention
                 route_time_benchmarks_minutes = (route_time_benchmarks_seconds / 60).rename('Route Benchmark Duration (min)')

                 # Merge the benchmark (in minutes) back
                 self.df = self.df.merge(
                     route_time_benchmarks_minutes,
                     on=['Startplats', 'Slutplats'],
                     how='left'
                 )
                 # Fill potentially missing benchmarks (for routes with no positive durations) with NaN
                 self.df['Route Benchmark Duration (min)'].fillna(np.nan, inplace=True)

                 # Convert the benchmark back to timedelta for 'Real Start Time' calculation
                 self.df['Route Benchmark Duration Tdelta'] = pd.to_timedelta(self.df['Route Benchmark Duration (min)'], unit='m')
                 # Handle NaNs that couldn't be converted
                 self.df['Route Benchmark Duration Tdelta'].fillna(pd.NaT, inplace=True)
            else:
                 print("Warning: No positive durations found to calculate any route benchmarks.")
                 self.df['Route Benchmark Duration (min)'] = np.nan
                 self.df['Route Benchmark Duration Tdelta'] = pd.NaT

            # --- Calculate Real Start Time ---
            # Real Start Time = Uppdrag Sluttid - Benchmark Duration for that route
            print("Calculating Real Start Time...")
            if 'Route Benchmark Duration Tdelta' in self.df.columns:
                 self.df['Real Start Time'] = self.df['Uppdrag Sluttid'] - self.df['Route Benchmark Duration Tdelta']
                 # Fill NaT resulting from NaT benchmark
                 self.df['Real Start Time'].fillna(pd.NaT, inplace=True)
            else:
                 print("Warning: Benchmark column missing, cannot calculate Real Start Time.")
                 self.df['Real Start Time'] = pd.NaT

            # --- Calculate Real Waiting Time ---
            print("Calculating Real Waiting Time...")
            if 'Real Start Time' in self.df.columns and 'Skapad Tid' in self.df.columns: # Need Skapad Tid too
                # Ensure 'Real Start Time' is datetime before subtraction
                if pd.api.types.is_datetime64_any_dtype(self.df['Real Start Time']):
                    self.df['Real Waiting Time (minutes)'] = (self.df['Real Start Time'] - self.df['Skapad Tid']).dt.total_seconds() / 60
                    # Fill NaN resulting from NaT Real Start Time or calculation issues
                    self.df['Real Waiting Time (minutes)'].fillna(np.nan, inplace=True)
                else:
                    print("Warning: 'Real Start Time' is not datetime. Cannot calculate Real Waiting Time.")
                    self.df['Real Waiting Time (minutes)'] = np.nan
            else:
                 print("Warning: 'Real Start Time' or 'Skapad Tid' column missing, cannot calculate Real Waiting Time.")
                 self.df['Real Waiting Time (minutes)'] = np.nan


            # --- Add Hour Column ---
            print("Adding Hour column...")
            # Check if 'Uppdrag Sluttid' exists and is datetime
            if 'Uppdrag Sluttid' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['Uppdrag Sluttid']):
                 self.df['Hour'] = self.df['Uppdrag Sluttid'].dt.hour
            else:
                 print("Warning: Cannot create 'Hour' column ('Uppdrag Sluttid' missing or not datetime).")
                 self.df['Hour'] = np.nan # Add column as NaN if it can't be calculated


            # --- Filtering ---
            print("Filtering based on Real Waiting Time (0-300 min)...")
            if 'Real Waiting Time (minutes)' in self.df.columns:
                 initial_rows_filter = len(self.df)
                 # Also filter NaNs which would fail the comparison
                 self.df = self.df.dropna(subset=['Real Waiting Time (minutes)'])
                 self.df = self.df[
                     (self.df['Real Waiting Time (minutes)'] >= 0) &
                     (self.df['Real Waiting Time (minutes)'] <= 300)
                 ]
                 dropped_count = initial_rows_filter - len(self.df)
                 if dropped_count > 0:
                    print(f"Filtered {dropped_count} rows based on Real Waiting Time (0-300 min, non-NaN).")
            else:
                 print("Warning: Cannot filter on 'Real Waiting Time (minutes)' as it wasn't calculated or contains only NaNs.")


            # --- Final Checks ---
            if self.df.empty:
                print("Error: DataFrame became empty during preprocessing.")
                return None

            end_proc_time = time.time()
            print(f"--- Data Preprocessing Finished ({end_proc_time - start_proc_time:.2f} seconds) ---")
            print(f"Processed DataFrame shape: {self.df.shape}")
            # print("Columns in processed data:", self.df.columns.tolist()) # Uncomment for debugging
            return self.df

        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            return None


# ======================================
#        Analysis Classes
# ======================================

class GeneralAnalysis:
    def __init__(self, processed_df):
         """Initializes with the processed DataFrame."""
         if processed_df is None or processed_df.empty:
             raise ValueError("Input DataFrame for GeneralAnalysis is empty or None.")
         self.df = processed_df
         self.hourly_frequency_data = None # Store result

    def calculate_hourly_frequency(self):
        """Calculates the frequency of requests per hour based on 'Skapad Tid'."""
        if 'Skapad Tid' not in self.df.columns:
            print("Error (Hourly Freq): 'Skapad Tid' column not found.")
            self.hourly_frequency_data = None
            return None
        if not pd.api.types.is_datetime64_any_dtype(self.df['Skapad Tid']):
             print("Error (Hourly Freq): 'Skapad Tid' column is not in datetime format.")
             self.hourly_frequency_data = None
             return None

        df_copy = self.df.copy()
        df_copy['request_hour'] = df_copy['Skapad Tid'].dt.hour
        hourly_freq = df_copy['request_hour'].value_counts().sort_index().reset_index()
        hourly_freq.columns = ['Hour', 'RequestCount']

        # Ensure all hours 0-23 are present, filling missing with 0 count
        all_hours = pd.DataFrame({'Hour': range(24)})
        hourly_freq = pd.merge(all_hours, hourly_freq, on='Hour', how='left').fillna(0)
        hourly_freq['RequestCount'] = hourly_freq['RequestCount'].astype(int) # Ensure integer counts

        self.hourly_frequency_data = hourly_freq # Store it
        return self.hourly_frequency_data

    # --- NEW Plotting Function for Hourly Demand ---
    def plot_hourly_demand(self):
        """Plots the calculated hourly request demand."""
        if self.hourly_frequency_data is None:
            print("Hourly frequency data not calculated yet. Run calculate_hourly_frequency() first.")
            # Try to calculate it if not present
            if self.calculate_hourly_frequency() is None:
                 print("Cannot plot hourly demand, calculation failed.")
                 return

        if self.hourly_frequency_data.empty:
             print("No hourly frequency data to plot.")
             return

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Hour', y='RequestCount', data=self.hourly_frequency_data, color='skyblue')

        ax.set_title('Hourly Request Demand (Based on Skapad Tid)', fontsize=16)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Number of Requests', fontsize=12)
        ax.set_xticks(range(24)) # Ensure all hours are labeled
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show() # Show the plot immediately


    def calculate_start_end_frequency(self):
        """Calculates the frequency of requests per unique start-end location pair."""
        start_col = 'Startplats'
        end_col = 'Slutplats'
        group_cols = [start_col, end_col]

        if not all(col in self.df.columns for col in group_cols):
             print(f"Error (Start/End Freq): Missing columns {group_cols}.")
             return None

        df_copy = self.df.copy()
        df_copy[start_col] = df_copy[start_col].fillna('Unknown_Start_Name')
        df_copy[end_col] = df_copy[end_col].fillna('Unknown_End_Name')

        start_end_freq = df_copy.groupby(group_cols).size().reset_index(name='RequestCount')
        start_end_freq = start_end_freq.sort_values('RequestCount', ascending=False)
        return start_end_freq

    # --- MODIFIED Edge Weight Estimation ---
    def estimate_edge_weights(self):
        """Estimates travel time using the MEDIAN of 'Actual Transport Duration (minutes)'."""
        required_cols = ['Actual Transport Duration (minutes)', 'Startplats', 'Slutplats']
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            print(f"Error (Edge Weights): Missing columns {missing}.")
            return None

        # Ensure duration is numeric
        if not pd.api.types.is_numeric_dtype(self.df['Actual Transport Duration (minutes)']):
             print("Error (Edge Weights): 'Actual Transport Duration (minutes)' is not numeric.")
             return None

        df_copy = self.df.copy()
        # Drop rows where duration is NaN or non-positive for median calculation
        valid_durations = df_copy.dropna(subset=['Actual Transport Duration (minutes)'])
        valid_durations = valid_durations[valid_durations['Actual Transport Duration (minutes)'] > 0]

        if valid_durations.empty:
            print("Warning (Edge Weights): No valid positive durations found.")
            return pd.DataFrame(columns=['Startplats', 'Slutplats', 'Estimated_Travel_Time_Minutes (Median)', 'Sample_Size']) # Updated col name

        start_col = 'Startplats'
        end_col = 'Slutplats'
        group_cols = [start_col, end_col]
        display_cols = group_cols[:]

        valid_durations[start_col] = valid_durations[start_col].fillna('Unknown_Start_Name')
        valid_durations[end_col] = valid_durations[end_col].fillna('Unknown_End_Name')

        # --- Use 'median' instead of q10 ---
        edge_weights = valid_durations.groupby(group_cols)['Actual Transport Duration (minutes)'].agg(
            ['median', 'size'] # Calculate median and count
        ).reset_index()

        # --- Update renaming ---
        edge_weights.rename(columns={'median': 'Estimated_Travel_Time_Minutes (Median)',
                                     'size': 'Sample_Size'}, inplace=True)
        # --- Update sorting ---
        edge_weights = edge_weights.sort_values('Estimated_Travel_Time_Minutes (Median)')

        final_display_cols = [col for col in display_cols if col in edge_weights.columns]
        # --- Update final columns list ---
        return edge_weights[final_display_cols + ['Estimated_Travel_Time_Minutes (Median)', 'Sample_Size']]


class RouteDisplayTable:
    def __init__(self, processed_df):
        """Initializes with the processed DataFrame."""
        required_cols = ['Startplats', 'Slutplats', 'Actual Transport Duration (minutes)', 'Skapad Tid', 'Uppdrag Sluttid']
        if processed_df is None or processed_df.empty:
             raise ValueError("Input DataFrame for RouteDisplayTable is empty or None.")
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns for RouteDisplayTable: {missing_cols}")
        self.df = processed_df


    def get_table_data(self):
        """
        Groups the DataFrame by 'Startplats' and 'Slutplats' and computes metrics.
          - 'Actual Transport Duration (Median) (min)': Median of the
            'Actual Transport Duration (minutes)' for the route. <- CHANGED to MEDIAN
          - 'Median Raw Duration (Sluttid - Skapad Tid) (min)': Median of (Uppdrag Sluttid - Skapad Tid).
        """
        df_copy = self.df.copy() # Work on copy
        # Ensure Actual Transport Duration is numeric and drop NaNs for median
        df_copy['Actual Transport Duration (minutes)'] = pd.to_numeric(df_copy['Actual Transport Duration (minutes)'], errors='coerce')
        df_valid_actual = df_copy.dropna(subset=['Actual Transport Duration (minutes)'])

        # Group by route
        grouped = df_valid_actual.groupby(['Startplats', 'Slutplats'])
        table_rows = []
        for (start, stop), group in grouped:
            # --- Calculate MEDIAN of 'Actual Transport Duration (minutes)' ---
            median_actual_duration = group['Actual Transport Duration (minutes)'].median() if not group.empty else np.nan

            # Median of raw durations (Uppdrag Sluttid - Skapad Tid) for the *original* group
            original_group_slice = self.df[(self.df['Startplats'] == start) & (self.df['Slutplats'] == stop)]
            if not original_group_slice.empty:
                 raw_duration = (original_group_slice['Uppdrag Sluttid'] - original_group_slice['Skapad Tid']).dt.total_seconds() / 60
                 median_raw_duration = raw_duration.median() if not raw_duration.empty else np.nan
            else:
                 median_raw_duration = np.nan

            # --- Update row data ---
            table_rows.append([start, stop, median_actual_duration, median_raw_duration])

        # --- Update columns ---
        result_df = pd.DataFrame(
            table_rows,
            columns=[
                'Startplats',
                'Slutplats',
                'Actual Transport Duration (Median) (min)', # Updated name
                'Median Raw Duration (Sluttid - Skapad Tid) (min)'
            ]
        )
        # --- Update sorting ---
        result_df = result_df.sort_values('Actual Transport Duration (Median) (min)', ascending=True)
        return result_df

    def plot_table(self):
        """Uses matplotlib to display the route transport table."""
        data = self.get_table_data()
        if data is None or data.empty:
             print("No data to plot in route display table.")
             return
        fig, ax = plt.subplots(figsize=(14, max(3, len(data) * 0.3))) # Wider figure
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=data.values, colLabels=data.columns, loc='center', cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9) # Slightly smaller font for more columns
        the_table.scale(1.1, 1.6) # Adjust scaling
        plt.title("Route Transport Time Metrics", fontsize=14)
        plt.tight_layout() # Helps fit table
        plt.show()