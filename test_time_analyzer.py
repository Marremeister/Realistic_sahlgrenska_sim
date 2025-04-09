"""
Test script for the enhanced TransportDataAnalyzer with time range analysis.
This can be used to verify the functionality and generate example data.
"""
import os
import sys
import logging
from Model.Data_processor.transport_data_analyzer import TransportDataAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TimeBasedAnalysisTest")


def test_time_range_analysis(file_path, output_dir='analysis_output'):
    """
    Test the time-based analysis functionality.

    Args:
        file_path: Path to Excel/CSV file with transport data
        output_dir: Directory to save analysis results
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    logger.info(f"Initializing time-based data analyzer for {file_path}")
    analyzer = TransportDataAnalyzer(file_path)

    # Load and clean the data
    if not analyzer.load_data():
        logger.error("Failed to load data. Exiting.")
        return

    cleaned_data = analyzer.clean_data()
    if cleaned_data is None or cleaned_data.empty:
        logger.error("Failed to clean data. Exiting.")
        return

    # Export time-based statistics
    logger.info("Exporting time-based statistics...")
    analyzer.export_time_based_statistics(output_dir)

    # Test generating realistic requests for specific time ranges
    time_ranges = [
        (8, 10),  # Morning peak
        (12, 14),  # Lunch time
        (16, 18),  # Evening peak
        (22, 6)  # Night
    ]

    for start, end in time_ranges:
        logger.info(f"Generating requests for time range {start}-{end}...")

        # Generate 10 sample requests for each time range
        requests = analyzer.generate_realistic_requests(start, end, 10)

        # Log the generated requests
        logger.info(f"Generated {len(requests)} requests for time range {start}-{end}:")
        for i, (origin, dest, transport_type, urgent) in enumerate(requests):
            logger.info(f"  {i + 1}. {origin} → {dest} ({transport_type}, urgent={urgent})")

        # Save to JSON for later use
        import json
        with open(os.path.join(output_dir, f'sample_requests_{start:02d}_{end:02d}.json'), 'w') as f:
            json.dump([
                {
                    'origin': origin,
                    'destination': dest,
                    'transport_type': transport_type,
                    'urgent': urgent
                }
                for origin, dest, transport_type, urgent in requests
            ], f, indent=2)

    # Generate statistics for all 24 hours
    logger.info("Generating hourly statistics...")
    hourly_rates = {}
    for hour in range(24):
        next_hour = (hour + 1) % 24
        rate = analyzer.get_request_rate_by_time_range(hour, next_hour)
        hourly_rates[f"{hour:02d}-{next_hour:02d}"] = rate
        logger.info(f"Hour {hour:02d}-{next_hour:02d}: {rate:.2f} requests/hour")

    # Save hourly rates
    import json
    with open(os.path.join(output_dir, 'hourly_rates.json'), 'w') as f:
        json.dump(hourly_rates, f, indent=2)

    logger.info(f"Time analysis test complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_time_analyzer.py <path_to_excel_or_csv_file> [output_directory]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'analysis_output'

    test_time_range_analysis(file_path, output_dir)