"""
Example showing the complete workflow for time-based benchmarks:
1. Process and save analysis data (one-time operation)
2. Load data from files for benchmarking
3. Generate and run time-based benchmarks
"""
import os
import logging
import numpy as np
from Model.Data_processor.transport_data_analyzer import TransportDataAnalyzer
from benchmark.benchmark_model import BenchmarkModel
from benchmark.benchmark_plotter import BenchmarkAnalysis
from new_backend_benchmark.time_based_benchmark_generator import TimeBasedBenchmarkGenerator
from new_backend_benchmark.transport_data_repository import TransportDataRepository, enhance_transport_data_analyzer_export


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TimeBasedBenchmarkExample")


def process_transport_data(file_path, output_dir='analysis_output'):
    """
    Process transport data and save analysis results to files (one-time operation).
    """
    logger.info(f"Processing transport data from {file_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = TransportDataAnalyzer(file_path)

    # Load and clean data
    if not analyzer.load_data():
        logger.error("Failed to load data")
        return False

    if analyzer.clean_data() is None:
        logger.error("Failed to clean data")
        return False

    # Export time-based statistics
    logger.info("Exporting time-based statistics...")
    analyzer.export_time_based_statistics(output_dir)

    # Also export additional data needed by the repository
    enhance_transport_data_analyzer_export(analyzer, output_dir)

    logger.info(f"Processed data saved to {output_dir}")
    return True


def run_time_based_benchmarks(system, data_dir='analysis_output'):
    """
    Run time-based benchmarks using saved analysis data.
    """
    # Check if data exists
    if not TransportDataRepository.check_data_exists(data_dir):
        logger.error(f"Required data files not found in {data_dir}")
        return False

    # Initialize repository and generator
    repository = TransportDataRepository(data_dir)
    generator = TimeBasedBenchmarkGenerator(repository)

    # Initialize benchmark model
    benchmark_model = BenchmarkModel(system)

    # Define benchmark parameters
    time_ranges = [
        (8, 12),  # Morning
        (12, 17),  # Afternoon
        (17, 22),  # Evening
        (8, 17)  # Full workday
    ]

    transporter_counts = [3, 5, 10]

    # Run benchmarks for each combination
    results = {}

    for start_hour, end_hour in time_ranges:
        time_key = f"{start_hour:02d}-{end_hour:02d}"

        for num_transporters in transporter_counts:
            logger.info(f"Running benchmark for time range {time_key} with {num_transporters} transporters")

            # Generate scenario
            scenario = generator.generate_benchmark_scenario(start_hour, end_hour, num_transporters)

            # Save scenario
            generator.save_benchmark_scenario(scenario)

            # Run benchmark
            transporters, requests = generator.convert_scenario_to_benchmark_format(scenario)

            # Only run a few random iterations for the example
            optimal_times = benchmark_model.run_benchmark("ilp", 1, transporters, requests)
            random_times = benchmark_model.run_benchmark("random", 50, transporters, requests)

            # Get workload distributions
            workload_opt = benchmark_model.get_workload_distribution("ilp", transporters, requests)
            workload_rand = benchmark_model.get_workload_distribution("random", transporters, requests)

            # Calculate statistics
            opt_std = benchmark_model.calculate_workload_std(workload_opt)
            rand_std = benchmark_model.calculate_workload_std(workload_rand)

            # Log results
            logger.info(f"Results for {time_key} with {num_transporters} transporters:")
            logger.info(f"  Optimal time: {optimal_times[0]:.2f} seconds")
            logger.info(f"  Random mean: {np.mean(random_times):.2f} seconds")
            logger.info(
                f"  Improvement: {((np.mean(random_times) - optimal_times[0]) / np.mean(random_times) * 100):.2f}%")

            # Store results
            result_key = f"{time_key}_{num_transporters}t"
            results[result_key] = {
                "optimal_time": optimal_times[0],
                "random_times": random_times,
                "workload_opt": workload_opt,
                "workload_rand": workload_rand,
                "opt_std": opt_std,
                "rand_std": rand_std
            }

            # Create visualization (if running in an environment with plotting capabilities)
            try:
                result_data = {
                    f"{time_key} ({num_transporters} transporters)": random_times
                }
                optimal_ref = {
                    f"{time_key} ({num_transporters} transporters)": optimal_times[0]
                }

                view = BenchmarkAnalysis(result_data, optimal_ref)
                view.analyze_all()
                view.plot_side_by_side_workload(workload_opt, workload_rand)
            except Exception as e:
                logger.warning(f"Could not create visualization: {str(e)}")

    return results


def mock_system():
    """Create a mock system for standalone testing."""

    class MockSystem:
        def add_transporter(self, name):
            return name

        def create_transport_request(self, origin, dest, transport_type="stretcher", urgent=False):
            """Mock method to create a transport request."""
            request = MockRequest(origin, dest)
            MockRequest.pending_requests.append(request)
            return request

        class TransportManager:
            def get_transporter_objects(self):
                class MockTransporter:
                    def __init__(self, name):
                        self.name = name
                        self.current_location = "Reception"
                        self.pathfinder = MockPathfinder()

                return [MockTransporter(f"T{i}") for i in range(5)]

            class Strategy:
                def generate_assignment_plan(self, transporters, requests, graph):
                    # Simple mock implementation
                    if not transporters or not requests:
                        return {}

                    result = {}
                    for i, t in enumerate(transporters):
                        result[t.name] = requests[i::len(transporters)]
                    return result

                def estimate_travel_time(self, transporter, request):
                    return 60  # Mock 60 seconds

            assignment_strategy = Strategy()

            def enable_random_mode(self):
                pass

            def enable_optimized_mode(self):
                pass

        transport_manager = TransportManager()

        class Hospital:
            def get_graph(self):
                class MockGraph:
                    def get_edge_weight(self, src, dst):
                        return 10  # Mock weight

                return MockGraph()

        hospital = Hospital()

    class MockPathfinder:
        def dijkstra(self, src, dst):
            # Simple mock implementation
            return [src, dst], 60

    class MockRequest:
        pending_requests = []
        ongoing_requests = []
        completed_requests = []

        def __init__(self, origin, destination):
            self.origin = origin
            self.destination = destination

    # Replace the actual TransportationRequest with the mock
    import sys
    from unittest.mock import MagicMock

    # This mocks the module import
    sys.modules['Model.model_transportation_request'] = MagicMock()
    sys.modules['Model.model_transportation_request'].TransportationRequest = MockRequest

    return MockSystem()


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example_time_benchmark.py <csv_file_path> [mode]")
        print("  Modes: process - Process data only")
        print("         benchmark - Run benchmarks only (requires processed data)")
        print("         all - Process data and run benchmarks (default)")
        sys.exit(1)

    file_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'all'

    # Process data if needed
    if mode in ['process', 'all']:
        process_transport_data(file_path)

    # Run benchmarks if needed
    if mode in ['benchmark', 'all']:
        # For standalone testing, use a mock system
        # In a real environment, use the actual hospital system
        system = mock_system()
        run_time_based_benchmarks(system)