import pandas as pd # Import pandas if helper functions need it directly (optional here)
import traceback   # Import traceback for detailed error printing

# Import analysis and utility classes
from analysverktyg import DataLoader, DataPreprocessor, RouteDisplayTable, GeneralAnalysis
from queue import QueueingAnalysis
from workload import HourlyWorkloadAnalysis
from plotters import Plotter
from scrollable_table import ScrollableTable

# Configuration (can be moved outside if preferred)
FILE_PATH = "Columna Sahlgrenska 2024 Anomymiserad data.csv" # << CHECK PATH
ROUTE_BENCHMARK_PERCENTAGE = 10

# ===========================================
# Helper Functions for Each Analysis Step
# ===========================================

def load_raw_data(file_path):
    """Loads the raw data using DataLoader."""
    print(f"--- Loading Raw Data from: {file_path} ---")
    data_loader = DataLoader(file_path)
    raw_df = data_loader.load_data()
    if raw_df is None:
        print("Failed to load raw data.")
    else:
        print(f"Raw data loaded successfully. Shape: {raw_df.shape}")
    return raw_df

def preprocess_data(raw_df, percentage):
    """Preprocesses the raw data using DataPreprocessor."""
    if raw_df is None:
        print("Cannot preprocess data, raw DataFrame is None.")
        return None

    print("\n--- Preprocessing Data ---")
    processed_df = None # Initialize
    try:
        preprocessor = DataPreprocessor(raw_df)
        processed_df = preprocessor.preprocess_data(route_percentage=percentage)
    except ValueError as ve:
         print(f"Error initializing preprocessor: {ve}")
    except Exception as e:
         print(f"An unexpected error occurred during preprocessing: {e}")
         traceback.print_exc()

    if processed_df is None or processed_df.empty:
        print("Data preprocessing failed or resulted in an empty DataFrame.")
        return None
    else:
        print(f"Data preprocessing finished. Processed DataFrame shape: {processed_df.shape}")
        return processed_df

def run_general_analysis(processed_df):
    """Runs and prints/plots results for General Analysis.""" # Updated docstring
    if processed_df is None:
        print("Skipping General Analysis, processed DataFrame is None.")
        return None # Return None if analysis can't run

    print("\n--- Performing General Analysis ---")
    general_analyzer = None # Initialize
    try:
        general_analyzer = GeneralAnalysis(processed_df)

        # --- Hourly Frequency ---
        hourly_freq_table = general_analyzer.calculate_hourly_frequency()
        if hourly_freq_table is not None:
            print("\n1. Hourly Request Frequency (based on Skapad Tid):")
            print(hourly_freq_table.to_string())
            # --- Plot Hourly Demand ---
            print("Plotting hourly demand graph...")
            general_analyzer.plot_hourly_demand() # Call the new plotting function
        else:
            print("Hourly frequency calculation failed.")


        # --- Start-End Frequency ---
        start_end_freq_table = general_analyzer.calculate_start_end_frequency()
        if start_end_freq_table is not None:
            print("\n2. Start-End Location Frequency (Top 20):")
            print(start_end_freq_table.head(20).to_string())
        else:
             print("Start-End frequency calculation failed.")

        # --- Edge Weights (Now Median) ---
        edge_weights_table = general_analyzer.estimate_edge_weights()
        if edge_weights_table is not None:
            # --- Updated Print Statement ---
            print("\n3. Estimated Edge Weights (Fastest 20 Routes - MEDIAN Travel Time):")
            print(edge_weights_table.head(20).to_string())
        else:
            print("Edge weight estimation failed.")

        return general_analyzer # Return the analyzer instance if needed later

    except ValueError as ve:
        print(f"Error initializing GeneralAnalysis: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during General Analysis: {e}")
        traceback.print_exc()
    return general_analyzer # Return even if errors occurred during processing

def run_workload_analysis(processed_df):
    """Runs workload analysis and displays plots."""
    if processed_df is None:
        print("Skipping Workload Analysis, processed DataFrame is None.")
        return

    print("\n--- Performing Workload Analysis ---")
    try:
        workload_analysis = HourlyWorkloadAnalysis(processed_df)
        workload_analysis.calculate_workload_data() # Calculate internal data

        if workload_analysis.hourly_data:
            print("Plotting hourly workload histograms...")
            plotter1 = Plotter()
            workload_analysis.plot_hourly_workload(plotter=plotter1)
            plotter1.show_plots() # Shows the plot window

            print("Plotting transporter workload bar charts...")
            plotter2 = Plotter()
            workload_analysis.plot_transporter_workload_by_hour(plotter=plotter2)
            plotter2.show_plots() # Shows the plot window
        else:
            print("Skipping workload plots: No valid hourly data calculated.")

    except ValueError as ve:
        print(f"Error initializing HourlyWorkloadAnalysis: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during Workload Analysis: {e}")
        traceback.print_exc()

def run_queueing_analysis(processed_df):
    """Runs queueing analysis and displays plots."""
    if processed_df is None:
        print("Skipping Queueing Analysis, processed DataFrame is None.")
        return

    print("\n--- Performing Queueing Analysis ---")
    try:
        queueing_analysis = QueueingAnalysis(processed_df)
        print("Displaying Recalculated Queueing Analysis plot...")
        queueing_analysis.queueing_analysis()  # Recalculated - Shows the plot window
        print("Displaying Raw Queueing Analysis plot...")
        queueing_analysis.queueing_analysis_raw() # Raw - Shows the plot window
    except ValueError as ve:
        print(f"Error initializing QueueingAnalysis: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during Queueing Analysis: {e}")
        traceback.print_exc()

def display_route_table(processed_df):
    """Calculates and displays the route display table."""
    if processed_df is None:
        print("Skipping Route Table Display, processed DataFrame is None.")
        return

    print("\n--- Preparing Route Display Table ---")
    try:
        route_table_display = RouteDisplayTable(processed_df)
        table_df = route_table_display.get_table_data()

        if table_df is not None and not table_df.empty:
            print(f"Route display table created with {len(table_df)} rows.")
            print("Displaying route table in scrollable window...")
            scroll_table = ScrollableTable(table_df)
            scroll_table.show() # Shows the Tkinter window
        else:
            print("Route display table data is empty or calculation failed.")

    except ValueError as ve:
        print(f"Error initializing RouteDisplayTable: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during Route Table display: {e}")
        traceback.print_exc()


# ===========================================
# Main Orchestration Function
# ===========================================

def run_analysis_pipeline():
    """Runs the complete data analysis pipeline."""

    # 1. Load
    raw_df = load_raw_data(FILE_PATH)
    if raw_df is None: return

    # 2. Preprocess
    processed_df = preprocess_data(raw_df, ROUTE_BENCHMARK_PERCENTAGE)
    if processed_df is None: return

    # 3. Run Analyses (pass the processed data to each)
    # Store the general_analyzer instance to potentially reuse its calculated data
    general_analyzer_instance = run_general_analysis(processed_df)
    run_workload_analysis(processed_df)
    run_queueing_analysis(processed_df)
    display_route_table(processed_df)

    print("\n--- Analysis Run Finished ---")


# ===========================================
# Entry Point
# ===========================================

if __name__ == "__main__":
    run_analysis_pipeline() # Call the main pipeline function