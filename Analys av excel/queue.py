
import matplotlib.pyplot as plt
import numpy as np # For NaN handling

class QueueingAnalysis:
    def __init__(self, processed_df): # Expects processed_df now
        """Initializes with the processed DataFrame."""
        if processed_df is None or processed_df.empty:
             raise ValueError("Input DataFrame for QueueingAnalysis is empty or None.")
         # Check for columns needed by EITHER raw or recalculated analysis
        required_cols = ['Real Waiting Time (minutes)', 'Skapad Tid', 'Uppdrag Sluttid']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
             # Raise error only if *both* analysis types would fail
             # If only 'Real Waiting Time' is missing, raw analysis might still work.
             if 'Real Waiting Time (minutes)' in missing_cols and \
                ('Skapad Tid' in missing_cols or 'Uppdrag Sluttid' in missing_cols):
                 raise ValueError(f"DataFrame missing essential columns for Queueing Analysis: {missing_cols}")
             else:
                  print(f"Warning (Queueing): Missing columns {missing_cols}. Some analysis might not run.")

        self.df = processed_df

    # --- Plotting Helper Functions (remain the same) ---
    def plot_main_histogram(self, waiting_times, ax, label_prefix=""):
        # ... (no changes needed here) ...
        mean_wait = waiting_times.mean()
        median_wait = waiting_times.median()

        ax.hist(waiting_times, bins=range(0, int(waiting_times.max()) + 10, 10), edgecolor='black', alpha=0.7) # Dynamic bins
        ax.axvline(mean_wait, color='red', linestyle='dashed', linewidth=2,
                   label=f'{label_prefix}Mean: {mean_wait:.1f}')
        ax.axvline(median_wait, color='green', linestyle='dashed', linewidth=2,
                   label=f'{label_prefix}Median: {median_wait:.1f}')
        ax.set_xlabel(f'{label_prefix}Waiting Time (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of {label_prefix}Waiting Times')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)


    def plot_cdf(self, waiting_times, ax, label_prefix=""):
        # ... (no changes needed here) ...
        sorted_wait = waiting_times.sort_values()
        cdf = np.arange(1, len(sorted_wait) + 1) / len(sorted_wait) # More standard CDF calculation
        ax.plot(sorted_wait, cdf, marker='.', markersize=2, linestyle='none') # smaller markers
        ax.set_xlabel(f'{label_prefix}Waiting Time (minutes)')
        ax.set_ylabel('CDF')
        ax.set_title(f'CDF of {label_prefix}Waiting Times')
        ax.grid(True)


    def plot_summary_metrics(self, waiting_times, ax, label_prefix=""):
        # ... (no changes needed here) ...
        mean_wait = waiting_times.mean()
        median_wait = waiting_times.median()
        std_wait = waiting_times.std()
        max_wait = waiting_times.max()
        p90 = waiting_times.quantile(0.90)
        p95 = waiting_times.quantile(0.95)
        p99 = waiting_times.quantile(0.99)

        metrics = ['Mean', 'Median', 'Std', 'Max', '90th', '95th', '99th']
        values = [mean_wait, median_wait, std_wait, max_wait, p90, p95, p99]

        bars = ax.bar(metrics, values, color='skyblue')
        ax.set_title(f'Summary Metrics ({label_prefix.strip()})')
        ax.set_ylabel('Minutes')
        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom', ha='center') # Add text label

    # --- Main Analysis Functions ---
    def detailed_queueing_analysis(self):
        """
        Create a detailed analysis figure using recalculated waiting times
        ('Real Waiting Time (minutes)' column from the processed DataFrame).
        """
        print("Running detailed queueing analysis (Recalculated)...")
        if 'Real Waiting Time (minutes)' not in self.df.columns:
            print("Skipping Recalculated Queueing Analysis: 'Real Waiting Time (minutes)' column missing.")
            return

        # Use the pre-calculated column, filter NaNs and negatives (should be done in preproc, but double-check)
        waiting_times = self.df['Real Waiting Time (minutes)'].dropna()
        waiting_times = waiting_times[waiting_times >= 0] # Should already be filtered by preprocessor

        if waiting_times.empty:
             print("No valid recalculated waiting times found to analyze.")
             return

        print(f"Analyzing {len(waiting_times)} valid recalculated waiting time entries.")
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        ax_main = fig.add_subplot(gs[0, :])
        self.plot_main_histogram(waiting_times, ax_main, label_prefix="Recalculated ")

        ax_cdf = fig.add_subplot(gs[1, 0])
        self.plot_cdf(waiting_times, ax_cdf, label_prefix="Recalculated ")

        ax_summary = fig.add_subplot(gs[1, 1])
        self.plot_summary_metrics(waiting_times, ax_summary, label_prefix="Recalculated ")

        fig.suptitle("Queueing Analysis (Recalculated Waiting Time)", fontsize=16) # Add overall title
        plt.show()

    def detailed_queueing_analysis_raw(self):
        """
        Create a detailed analysis figure using raw waiting times calculated as
        (Uppdrag Sluttid - Skapad Tid), removing values above 300 minutes.
        """
        print("Running detailed queueing analysis (Raw)...")
        required_cols = ['Skapad Tid', 'Uppdrag Sluttid']
        if not all(col in self.df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in self.df.columns]
            print(f"Skipping Raw Queueing Analysis: Missing columns {missing}.")
            return

        # Calculate raw waiting time directly from processed_df columns
        raw_waiting_time = (self.df['Uppdrag Sluttid'] - self.df['Skapad Tid']).dt.total_seconds() / 60

        # Filter out NaNs, negative values, and values above 300 minutes
        raw_waiting_time = raw_waiting_time.dropna()
        raw_waiting_time = raw_waiting_time[(raw_waiting_time >= 0) & (raw_waiting_time <= 300)]

        if raw_waiting_time.empty:
             print("No valid raw waiting times (0-300 min) found to analyze.")
             return

        print(f"Analyzing {len(raw_waiting_time)} valid raw waiting time entries.")
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        ax_main = fig.add_subplot(gs[0, :])
        self.plot_main_histogram(raw_waiting_time, ax_main, label_prefix="Raw ")

        ax_cdf = fig.add_subplot(gs[1, 0])
        self.plot_cdf(raw_waiting_time, ax_cdf, label_prefix="Raw ")

        ax_summary = fig.add_subplot(gs[1, 1])
        self.plot_summary_metrics(raw_waiting_time, ax_summary, label_prefix="Raw ")

        fig.suptitle("Queueing Analysis (Raw Waiting Time: Sluttid - Skapad Tid)", fontsize=16)
        plt.show()

    # Aliases remain the same
    def queueing_analysis(self):
        self.detailed_queueing_analysis()

    def queueing_analysis_raw(self):
        self.detailed_queueing_analysis_raw()