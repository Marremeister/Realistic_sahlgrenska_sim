#!/usr/bin/env python3
"""
Script to analyze transport data and generate hourly statistics.
This analyzes a CSV file of transport data and produces JSON files with
hourly origin-destination patterns for use in benchmark generation.

Usage:
    python run_time_analyzer.py <path_to_csv_file> [output_directory]
"""
import os
import sys
import logging
import matplotlib.pyplot as plt
import pandas as pd
from Model.Data_processor.time_based_transport_analyzer import TimeBasedTransportAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TimeAnalysis")


def run_time_analysis(file_path, output_dir='analysis_output'):
    """
    Run time-based analysis on transport data.

    Args:
        file_path (str): Path to CSV or Excel file with transport data
        output_dir (str): Directory to save analysis results

    Returns:
        bool: True if analysis was successful
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    logger.info(f"Initializing time-based analyzer for {file_path}")
    analyzer = TimeBasedTransportAnalyzer(file_path, output_dir)

    # Run complete analysis
    if not analyzer.analyze_and_export():
        logger.error("Analysis failed.")
        return False

    # Generate summary visualizations
    logger.info("Generating summary visualizations...")
    generate_visualizations(analyzer, output_dir)

    logger.info(f"Time-based analysis complete. Results saved to {output_dir}")
    return True


def generate_visualizations(analyzer, output_dir):
    """
    Generate summary visualizations of the time-based analysis.

    Args:
        analyzer: Initialized TimeBasedTransportAnalyzer with results
        output_dir: Directory to save visualizations
    """
    # 1. Create hourly request distribution chart
    hourly_stats = analyzer.get_hourly_stats()
    hourly_counts = {}

    for hour_key, stats in hourly_stats.items():
        # Extract the starting hour from the hour key (e.g., "08-09" -> 8)
        hour = int(hour_key.split('-')[0])
        hourly_counts[hour] = stats.get('request_count', 0)

    # Sort by hour
    sorted_hours = sorted(hourly_counts.keys())
    counts = [hourly_counts.get(hour, 0) for hour in sorted_hours]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_hours, counts, color='#4b6cb7')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Requests')
    plt.title('Transport Requests by Hour of Day')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_request_distribution.png'))
    plt.close()

    # 2. Create time range comparison chart
    time_range_stats = analyzer.get_time_range_stats()
    ranges = list(time_range_stats.keys())
    requests_per_hour = [stats.get('requests_per_hour', 0) for stats in time_range_stats.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(ranges, requests_per_hour, color='#2c3e50')
    plt.xlabel('Time Range')
    plt.ylabel('Requests per Hour')
    plt.title('Average Requests per Hour by Time Range')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_range_comparison.png'))
    plt.close()

    # 3. Create top OD pairs visualization for a busy hour
    # Find the busiest hour
    busiest_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
    hour_key = f"{busiest_hour:02d}-{(busiest_hour + 1) % 24:02d}"
    busiest_hour_stats = hourly_stats.get(hour_key, {})

    # Get top 10 OD pairs
    od_pairs = busiest_hour_stats.get('od_pairs', [])[:10]  # Top 10

    if od_pairs:
        pairs = [f"{pair['origin']} → {pair['destination']}" for pair in od_pairs]
        probabilities = [pair['probability'] * 100 for pair in od_pairs]  # Convert to percentage

        plt.figure(figsize=(12, 8))
        plt.barh(pairs, probabilities, color='#3498db')
        plt.xlabel('Percentage of Requests')
        plt.ylabel('Origin → Destination')
        plt.title(f'Top Origin-Destination Pairs at {hour_key}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_od_pairs_hour_{busiest_hour}.png'))
        plt.close()


def print_usage():
    """Print script usage information."""
    print("Usage:")
    print("  python run_time_analyzer.py <path_to_csv_file> [output_directory]")
    print("\nDescription:")
    print("  Analyzes a CSV file of transport data and generates hourly statistics.")
    print("  The output includes JSON files with hourly origin-destination patterns.")
    print("  These can be used for generating realistic benchmark scenarios.")
    print("\nDefault output directory is 'analysis_output' (same as graph builder)")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Error: Missing required CSV file argument.")
        print_usage()
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'analysis_output'

    # Verify file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    # Run analysis
    if run_time_analysis(file_path, output_dir):
        print(f"\nTime-based analysis complete!")
        print(f"Results saved to: {os.path.abspath(output_dir)}")
    else:
        print("\nAnalysis failed. Check the logs for details.")
        sys.exit(1)