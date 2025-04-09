# run_data_processor.py
"""
Script to analyze transport data from Excel file and build a hospital graph.
"""
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Model.Data_processor import TransportDataAnalyzer, HospitalGraphBuilder, CoordinateGenerator
from Model.hospital_model import Hospital

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TransportDataAnalysis")


def analyze_data(file_path, output_dir='analysis_output'):
    """
    Analyze transport data from Excel file and generate a hospital graph.

    Args:
        file_path: Path to Excel file with transport data
        output_dir: Directory to save analysis results

    Returns:
        tuple: (hospital_graph, analyzer, builder) for further use
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    logger.info(f"Initializing data analyzer for {file_path}")
    analyzer = TransportDataAnalyzer(file_path)

    # Load and clean the data
    if not analyzer.load_data():
        logger.error("Failed to load data. Exiting.")
        return None, None, None

    cleaned_data = analyzer.clean_data()
    if cleaned_data is None or cleaned_data.empty:
        logger.error("Failed to clean data. Exiting.")
        return None, None, None

    # Generate analysis visualizations
    logger.info("Generating analysis visualizations...")

    # 1. Hourly request distribution
    hourly_dist = analyzer.get_hourly_request_distribution()
    _plot_hourly_distribution(hourly_dist, os.path.join(output_dir, 'hourly_distribution.png'))

    # 2. Transport frequency matrix
    freq_matrix = analyzer.get_frequency_matrix()
    if freq_matrix is not None:
        _plot_frequency_matrix(freq_matrix, os.path.join(output_dir, 'transport_frequency.png'))

    # 3. Create origin-destination frequency file
    _export_origin_destination_pairs(analyzer, os.path.join(output_dir, 'od_pairs.csv'))

    # Build hospital graph
    logger.info("Building hospital graph...")
    builder = HospitalGraphBuilder(analyzer, time_factor=0.1)  # Scale times to make simulation faster

    # Call build_graph with the parameters it accepts (min_weight and max_weight)
    graph = builder.build_graph(min_weight=0.5, max_weight=5.0)

    if hasattr(builder, 'name_mapping'):
        analyzer.name_mapping = builder.name_mapping

    # Ensure graph is connected
    builder.validate_graph_connectivity()

    # Generate coordinates with enhanced parameters
    logger.info("Generating improved coordinates for graph nodes...")

    # Step 1: Initialize the coordinate generator with larger canvas
    coord_generator = CoordinateGenerator(graph, canvas_width=1600, canvas_height=1200)

    # Step 2: Try force-directed layout with enhanced parameters
    try:
        logger.info("Applying force-directed layout with enhanced parameters...")
        coord_generator.generate_coordinates(
            iterations=2000,  # More iterations for better convergence
            repulsion_force=5.0,  # Stronger repulsion between nodes
            temperature=0.2,  # Higher temperature allows more movement
            cooling_factor=0.99  # Slower cooling maintains movement longer
        )
    except Exception as e:
        logger.warning(f"Force-directed layout failed: {str(e)}")
        logger.info("Falling back to grid layout...")
        coord_generator.generate_grid_layout(spacing=180)  # Increased spacing

    # Step 3: Adjust coordinates by department type
    try:
        logger.info("Adjusting coordinates by department type...")
        coord_generator.adjust_coordinates_by_department_type()
    except Exception as e:
        logger.warning(f"Department type adjustment failed: {str(e)}")

    # Step 4: Apply jitter to break up any remaining patterns
    logger.info("Applying coordinate jitter...")
    coord_generator.apply_jitter(amount=45)

    # Step 5: Ensure minimum distance between nodes if the method exists
    try:
        logger.info("Ensuring minimum distance between nodes...")
        if hasattr(coord_generator, 'ensure_minimum_distance'):
            coord_generator.ensure_minimum_distance(min_distance=80)
        else:
            logger.warning("ensure_minimum_distance method not available in current version")
    except Exception as e:
        logger.warning(f"Minimum distance enforcement failed: {str(e)}")

    # Step 6: Scale the layout to fit the canvas with good padding
    logger.info("Scaling layout to fit canvas...")
    coord_generator.scale_layout(padding=70)

    # Step 7: Final spreading of nodes for more space if the method exists
    try:
        logger.info("Applying final node spreading...")
        if hasattr(coord_generator, 'spread_nodes'):
            coord_generator.spread_nodes(expansion_factor=1.15)
        else:
            logger.warning("spread_nodes method not available in current version")
    except Exception as e:
        logger.warning(f"Node spreading failed: {str(e)}")

    # Step 8: Add transport lounge node positioned far from other nodes
    # We'll do this after all other operations to ensure it stays far from other nodes
    try:
        logger.info("Adding transport lounge node...")
        # Store the existing coordinates before adding the transport lounge
        existing_nodes = list(graph.adjacency_list.keys())

        # Find the maximum coordinates to position Transport Lounge far away
        max_x = max_y = 0
        for node in existing_nodes:
            try:
                x, y = graph.get_node_coordinates(node)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            except Exception:
                continue

        # Add the Transport Lounge node
        transport_node = "Transport Lounge"
        if transport_node not in graph.adjacency_list:
            graph.add_node(transport_node)

        # Position it far from other nodes (in the bottom right)
        offset = 300  # Large offset to ensure it's far away
        lounge_x = max_x + offset
        lounge_y = max_y + offset

        # Make sure it's within canvas bounds
        lounge_x = min(lounge_x, coord_generator.canvas_width - 100)
        lounge_y = min(lounge_y, coord_generator.canvas_height - 100)

        graph.set_node_coordinates(transport_node, lounge_x, lounge_y)

        # Connect Transport Lounge to key locations with realistic transport times
        major_nodes = ["Main Entrance", "Emergency Department", "Reception",
                       "Huvudentrén", "Akuten", "Mottagning"]

        connected = False
        for node in major_nodes:
            if node in graph.adjacency_list:
                graph.add_edge(transport_node, node, weight=30)  # 30 minutes transport time
                graph.add_edge(node, transport_node, weight=30)
                logger.info(f"Connected Transport Lounge to {node}")
                connected = True
                break

        # If no specific major node found, connect to the first node
        if not connected and existing_nodes:
            first_node = existing_nodes[0]
            graph.add_edge(transport_node, first_node, weight=30)
            graph.add_edge(first_node, transport_node, weight=30)
            logger.info(f"Connected Transport Lounge to {first_node}")

        logger.info(f"Added Transport Lounge at position ({lounge_x:.1f}, {lounge_y:.1f})")
    except Exception as e:
        logger.warning(f"Failed to add transport lounge: {str(e)}")

    # Export coordinates
    coord_file = os.path.join(output_dir, 'node_coordinates.json')
    logger.info(f"Exporting node coordinates to {coord_file}")
    coord_generator.export_coordinates(coord_file)

    # Create a Hospital instance with the graph
    hospital = Hospital()
    hospital.graph = graph

    logger.info(f"Analysis complete. Results saved to {output_dir}")

    if hospital and hospital.graph:
        graph_file = save_hospital_graph_for_integration(hospital)
        logger.info(f"Hospital graph saved for integration: {graph_file}")

    return hospital, analyzer, builder


def _plot_hourly_distribution(hourly_dist, output_file):
    """Plot hourly distribution of transport requests."""
    plt.figure(figsize=(12, 6))

    # Sort by hour
    hours = sorted(hourly_dist.keys())
    counts = [hourly_dist[h] for h in hours]

    # Plot
    plt.bar(hours, counts, color='#3498db')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Requests')
    plt.title('Transport Requests by Hour')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def _plot_frequency_matrix(freq_matrix, output_file):
    """Plot heatmap of transport frequencies between departments."""
    plt.figure(figsize=(14, 12))

    # Use seaborn for better heatmap
    mask = freq_matrix == 0  # Mask cells with zero frequency
    ax = sns.heatmap(freq_matrix, cmap="YlGnBu", linewidths=0.5,
                     cbar_kws={"shrink": 0.8}, mask=mask)

    plt.title('Transport Frequency Between Departments')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def _export_origin_destination_pairs(analyzer, output_file):
    """Export origin-destination pairs with frequencies and median times."""
    median_times = analyzer.get_median_transport_times()

    # Convert to DataFrame for easier manipulation
    data = []
    for (origin, dest), time in median_times.items():
        data.append({
            'Origin': origin,
            'Destination': dest,
            'MedianTimeSeconds': time
        })

    df = pd.DataFrame(data)

    # Count occurrences from cleaned data
    origin_col = analyzer._find_column_containing('origin', analyzer.cleaned_data.columns)
    dest_col = analyzer._find_column_containing('destination', analyzer.cleaned_data.columns)

    if origin_col and dest_col:
        count_df = analyzer.cleaned_data.groupby([origin_col, dest_col]).size().reset_index()
        count_df.columns = ['Origin', 'Destination', 'Frequency']

        # Merge with median times
        result = pd.merge(df, count_df, on=['Origin', 'Destination'], how='left')
        result['Frequency'] = result['Frequency'].fillna(0)
    else:
        # No frequency data available
        result = df
        result['Frequency'] = 0

    # Sort by frequency (descending)
    result = result.sort_values('Frequency', ascending=False)

    # Save to CSV
    result.to_csv(output_file, index=False)


def initialize_hospital_from_data(file_path, output_dir='analysis_output'):
    """
    Initialize a hospital from transport data file.

    Args:
        file_path: Path to Excel file with transport data
        output_dir: Directory to save analysis results

    Returns:
        Hospital: Hospital instance with graph populated from data
    """
    hospital, _, _ = analyze_data(file_path, output_dir)
    return hospital


def save_hospital_graph_for_integration(hospital, output_file='analysis_output/hospital_graph.json'):
    """Save the hospital graph in a format that can be loaded by the system."""
    import json

    # Extract data from hospital
    departments = hospital.graph.get_nodes()

    # Extract coordinates
    coordinates = {}
    for dept in departments:
        x, y = hospital.graph.get_node_coordinates(dept)
        coordinates[dept] = {'x': x, 'y': y}

    # Extract corridors
    corridors = []
    for source in hospital.graph.adjacency_list:
        for target, weight in hospital.graph.adjacency_list[source].items():
            # Only add each corridor once (since graph is undirected)
            if source < target:  # Simple way to ensure uniqueness
                corridors.append([source, target, weight])

    # Create data structure
    data = {
        'departments': departments,
        'coordinates': coordinates,
        'corridors': corridors
    }

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Hospital graph saved to {output_file}")
    return output_file


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_data_processor.py <path_to_excel_file> [output_directory]")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'analysis_output'

    # Run analysis
    hospital, analyzer, builder = analyze_data(file_path, output_dir)

    # Print summary
    if hospital and hospital.graph:
        print("\nGraph Summary:")
        print(f"  Nodes: {len(hospital.graph.get_nodes())}")
        print(
            f"  Edges: {sum(len(edges) for edges in hospital.graph.adjacency_list.values()) // 2}")  # Divide by 2 for undirected graph

        # Print some sample paths
        nodes = hospital.graph.get_nodes()
        if len(nodes) >= 2:
            from Model.model_pathfinder import Pathfinder

            pathfinder = Pathfinder(hospital)

            print("\nSample Paths:")
            for _ in range(3):
                source = nodes[0]
                target = nodes[-1]

                try:
                    path, distance = pathfinder.dijkstra(source, target)
                    print(f"  Path from {source} to {target}:")
                    print(f"    Distance: {distance:.2f} seconds")
                    print(f"    Path: {' -> '.join(path)}")
                except Exception as e:
                    print(f"  Error finding path: {str(e)}")