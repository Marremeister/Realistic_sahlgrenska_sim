# Model/data_processor/graph_builder.py
"""
Class for building a hospital graph based on transport data analysis.
"""
import logging
import random
from Model.graph_model import Graph
from Model.model_pathfinder import Pathfinder
from Model.Data_processor.department_name_normalizer import DepartmentNameNormalizer


class HospitalGraphBuilder:
    """
    Builds a hospital graph from analyzed transport data, using an incremental
    approach to create realistic connections.
    """

    def __init__(self, analyzer, time_factor=1.0):
        """
        Initialize the HospitalGraphBuilder.

        Args:
            analyzer: TransportDataAnalyzer instance with processed data
            time_factor: Factor to scale all times by (e.g., 0.1 for faster simulation)
        """
        self.analyzer = analyzer
        self.time_factor = time_factor
        self.graph = Graph(directed=False)
        self.logger = self._setup_logger()
        self.name_mapping = {}  # Will store mapping from original names to normalized names

    def _setup_logger(self):
        """Set up a logger for the HospitalGraphBuilder."""
        logger = logging.getLogger("HospitalGraphBuilder")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def build_graph(self, min_weight=0.5, max_weight=5.0):
        """
        Build a connected hospital graph based on transport data.

        Args:
            min_weight: Minimum edge weight
            max_weight: Maximum edge weight

        Returns:
            Graph: The constructed hospital graph
        """
        self.logger.info("Building connected hospital graph...")

        # Step 1: Get all departments and add as nodes
        all_departments = self.analyzer.get_all_departments()
        normalized_departments, self.name_mapping = self._normalize_department_names(all_departments)
        self._add_departments_as_nodes(normalized_departments)

        # Step 2: Get department pairs sorted by transport time
        sorted_pairs = self._prepare_department_pairs_normalized()

        # Step 3: Build a connected graph
        self._build_connected_graph_from_pairs(sorted_pairs, min_weight, max_weight)

        # Step 4: Add Transporter Lounge
        self._add_transporter_lounge()

        return self.graph

    def _build_connected_graph_from_pairs(self, sorted_pairs, min_weight, max_weight):
        """
        Build a connected graph from department pairs.

        Args:
            sorted_pairs: Sorted list of ((origin, dest), time) tuples
            min_weight: Minimum edge weight
            max_weight: Maximum edge weight
        """
        # Keep track of connected nodes
        connected_nodes = set()

        # First pass: Create a connected skeleton from the data
        self.logger.info("Building initial connected skeleton...")

        for (origin, dest), time in sorted_pairs:
            # Skip self-connections
            if origin == dest:
                continue

            # Normalize weight
            weight = time * self.time_factor
            if weight < min_weight:
                weight = min_weight
            elif weight > max_weight:
                weight = max_weight

            # Add edge if either node isn't connected yet
            if origin not in connected_nodes or dest not in connected_nodes:
                self.graph.add_edge(origin, dest, weight)
                connected_nodes.add(origin)
                connected_nodes.add(dest)
                self.logger.debug(f"Added edge for connectivity: {origin} -> {dest} (weight: {weight:.2f})")

        # Check if we have unconnected nodes and connect them
        all_nodes = set(self.graph.adjacency_list.keys())
        unconnected = all_nodes - connected_nodes

        if unconnected:
            self.logger.warning(f"Found {len(unconnected)} unconnected nodes after first pass. Connecting them...")

            # For each unconnected node, add an edge to a connected node
            for node in unconnected:
                # Find closest connected node by name similarity as a simple heuristic
                # In a real implementation, you might use spatial proximity if available
                connected_node = next(iter(connected_nodes))  # Default

                # Add edge with maximum weight
                self.graph.add_edge(node, connected_node, max_weight)
                connected_nodes.add(node)
                self.logger.debug(f"Connected isolated node: {node} -> {connected_node}")

        # Second pass: Add more edges from the data for a richer graph
        # But avoid creating too many redundant edges
        edge_count = sum(len(neighbors) for neighbors in self.graph.adjacency_list.values()) // 2
        target_edge_count = min(len(all_nodes) * 2, edge_count + len(sorted_pairs) // 2)

        self.logger.info(f"Enriching graph with additional edges (target: {target_edge_count})...")

        for (origin, dest), time in sorted_pairs:
            # Skip if we already have enough edges
            if edge_count >= target_edge_count:
                break

            # Skip self-connections or if edge already exists
            if origin == dest or dest in self.graph.adjacency_list[origin]:
                continue

            # Normalize weight
            weight = time * self.time_factor
            if weight < min_weight:
                weight = min_weight
            elif weight > max_weight:
                weight = max_weight

            # Add edge
            self.graph.add_edge(origin, dest, weight)
            edge_count += 1
            self.logger.debug(f"Added enrichment edge: {origin} -> {dest} (weight: {weight:.2f})")

        self.logger.info(f"Graph building complete. Graph has {len(all_nodes)} nodes and {edge_count} edges.")

    def _add_transporter_lounge(self):
        """Add a Transporter Lounge node to the graph."""
        if "Transporter Lounge" in self.graph.adjacency_list:
            self.logger.info("Transporter Lounge already exists in the graph.")
            return

        # Add Transporter Lounge node
        self.graph.add_node("Transporter Lounge")

        # Position it far from other nodes
        max_x = max_y = 0
        for node in self.graph.adjacency_list:
            try:
                x, y = self.graph.get_node_coordinates(node)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            except:
                pass

        # Position in the corner with some padding
        lounge_x = max_x + 200
        lounge_y = max_y + 200
        self.graph.set_node_coordinates("Transporter Lounge", lounge_x, lounge_y)

        # Connect to a central node (main entrance or first node)
        central_nodes = ["Huvudentrén", "Reception", "Vakten"]
        for node in central_nodes:
            if node in self.graph.adjacency_list:
                self.graph.add_edge("Transporter Lounge", node, 5.0)  # Maximum weight
                self.logger.info(f"Connected Transporter Lounge to {node}")
                return

        # If no central node found, connect to the first node
        first_node = next(iter(self.graph.adjacency_list))
        self.graph.add_edge("Transporter Lounge", first_node, 5.0)
        self.logger.info(f"Connected Transporter Lounge to {first_node}")

    def _normalize_department_names(self, departments):
        """
        Normalize department names by grouping similar names together using shared normalizer.

        Args:
            departments: List of original department names

        Returns:
            tuple: (normalized_names, name_mapping)
                - normalized_names: List of unique normalized department names
                - name_mapping: Dictionary mapping original names to normalized names
        """
        self.logger.info("Normalizing department names...")

        # Use the shared normalizer
        normalizer = DepartmentNameNormalizer(self.analysis_output_dir if hasattr(self, 'analysis_output_dir') else 'analysis_output')

        # First try to load existing mapping
        normalizer.load_existing_mapping()

        # Then normalize our departments
        name_mapping = normalizer.normalize_departments(departments)

        # Extract normalized names
        normalized_names = list(set(name_mapping.values()))

        # Save the mapping for future use
        normalizer.save_mapping()

        self.logger.info(f"Reduced {len(departments)} department names to {len(normalized_names)} unique departments")

        return normalized_names, name_mapping

    def _add_departments_as_nodes(self, departments=None):
        """
        Add departments as nodes in the graph.

        Args:
            departments: List of department names. If None, fetches from analyzer.
        """
        if departments is None:
            departments = self.analyzer.get_all_departments()

        for dept in departments:
            self.graph.add_node(dept)

        self.logger.info(f"Added {len(departments)} nodes to the graph")

    def _prepare_department_pairs_normalized(self):
        """
        Prepare normalized department pairs sorted by transport time.

        Returns:
            list: Sorted list of ((origin, dest), time) tuples with normalized names
        """
        # Get median transport times between departments
        original_median_times = self.analyzer.get_median_transport_times()

        # Normalize department names in transport times
        normalized_median_times = {}

        for (orig_origin, orig_dest), time in original_median_times.items():
            # Map original names to normalized names
            norm_origin = self.name_mapping.get(orig_origin, orig_origin)
            norm_dest = self.name_mapping.get(orig_dest, orig_dest)

            # Skip if same department after normalization
            if norm_origin == norm_dest:
                continue

            # If there are multiple paths between the same normalized departments,
            # take the minimum time (fastest path)
            key = (norm_origin, norm_dest)
            if key in normalized_median_times:
                normalized_median_times[key] = min(normalized_median_times[key], time)
            else:
                normalized_median_times[key] = time

        # Create a sorted list of department pairs by transport time
        return sorted(normalized_median_times.items(), key=lambda x: x[1])

    def _create_pathfinder(self):
        """
        Create a pathfinder for checking existing paths.

        Returns:
            Pathfinder: Initialized pathfinder object
        """

        # Create a temporary Hospital for the pathfinder
        class TempHospital:
            def __init__(self, graph):
                self.graph = graph

            def get_graph(self):
                return self.graph

        temp_hospital = TempHospital(self.graph)
        return Pathfinder(temp_hospital)

    def _add_edges_based_on_transport_data(self, sorted_pairs, pathfinder, path_threshold):
        """
        Add edges to the graph based on transport data and pathfinding.

        Args:
            sorted_pairs: Sorted list of ((origin, dest), time) tuples
            pathfinder: Pathfinder object for checking existing paths
            path_threshold: Threshold for adding direct edges
        """
        # Track connections that we've already examined
        examined_connections = set()
        added_edges = 0
        skipped_edges = 0

        # Iterate through pairs from shortest to longest time
        for (origin, dest), median_time in sorted_pairs:
            # Skip self-connections
            if origin == dest:
                continue

            # Skip if we've already examined this connection
            if (origin, dest) in examined_connections or (dest, origin) in examined_connections:
                continue

            examined_connections.add((origin, dest))

            # Scale the time by the factor
            scaled_time = median_time * self.time_factor

            try:
                # Check if there's already a path
                path, distance = pathfinder.dijkstra(origin, dest)

                if self._should_add_direct_edge(path, distance, scaled_time, path_threshold):
                    self.graph.add_edge(origin, dest, scaled_time)
                    added_edges += 1
                    self._log_edge_addition(origin, dest, scaled_time, path, distance)
                else:
                    # Existing path is good enough
                    skipped_edges += 1
                    self.logger.debug(f"Skipped edge {origin} -> {dest} - "
                                      f"Existing path is sufficient ({distance:.1f}s vs {scaled_time:.1f}s)")

            except Exception as e:
                # If there's an error in pathfinding, add direct edge to be safe
                self.logger.warning(f"Error in pathfinding from {origin} to {dest}: {str(e)}")
                self.graph.add_edge(origin, dest, scaled_time)
                added_edges += 1

        self.logger.info(f"Graph building complete. Added {added_edges} edges, skipped {skipped_edges} edges.")

    def _should_add_direct_edge(self, path, distance, scaled_time, path_threshold):
        """
        Determine if a direct edge should be added between departments.

        Args:
            path: Current path between departments
            distance: Current path distance
            scaled_time: Expected travel time
            path_threshold: Threshold for adding direct edges

        Returns:
            bool: True if direct edge should be added
        """
        # If no path exists, add direct edge
        if not path or len(path) < 2:
            return True

        # If path is significantly longer than expected, add direct edge
        if distance > scaled_time * path_threshold:
            return True

        return False

    def _log_edge_addition(self, origin, dest, scaled_time, path, distance):
        """Log the addition of an edge with appropriate reason."""
        if not path or len(path) < 2:
            self.logger.debug(f"Added edge {origin} -> {dest} (time: {scaled_time:.1f}s) - No existing path")
        else:
            self.logger.debug(f"Added edge {origin} -> {dest} (time: {scaled_time:.1f}s) - "
                              f"Existing path too long ({distance:.1f}s vs {scaled_time:.1f}s)")

    def add_edges_from_expert_input(self, edge_list):
        """
        Add or update edges based on expert input.

        Args:
            edge_list: List of tuples (origin, dest, time) to add/update

        Returns:
            int: Number of edges added/updated
        """
        added = 0
        for origin, dest, time in edge_list:
            # Normalize department names if mapping exists
            if hasattr(self, 'name_mapping'):
                origin = self.name_mapping.get(origin, origin)
                dest = self.name_mapping.get(dest, dest)

            if origin in self.graph.adjacency_list and dest in self.graph.adjacency_list:
                self.graph.add_edge(origin, dest, time * self.time_factor)
                added += 1
            else:
                self.logger.warning(f"Cannot add edge {origin} -> {dest}: Node(s) do not exist in graph")

        self.logger.info(f"Added/updated {added} edges from expert input")
        return added

    def validate_graph_connectivity(self):
        """
        Ensure the graph is fully connected by adding edges if necessary.

        Returns:
            bool: True if the graph was already connected, False if edges were added
        """
        self.logger.info("Validating graph connectivity...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.warning("No nodes in graph. Nothing to validate.")
            return True

        # Create a set of connected nodes, starting with the first node
        connected = {nodes[0]}
        frontier = [nodes[0]]

        # Breadth-first search to find all connected nodes
        while frontier:
            current = frontier.pop(0)
            neighbors = self.graph.adjacency_list[current].keys()

            for neighbor in neighbors:
                if neighbor not in connected:
                    connected.add(neighbor)
                    frontier.append(neighbor)

        # Check for disconnected nodes
        disconnected = set(nodes) - connected
        if not disconnected:
            self.logger.info("Graph is fully connected.")
            return True

        # Add edges to connect disconnected nodes
        self.logger.warning(f"Found {len(disconnected)} disconnected nodes. Adding connections...")

        for node in disconnected:
            # Find the closest connected node (by name similarity as a heuristic)
            connected_node = min(connected, key=lambda x: _name_similarity(node, x))

            # Add an edge with a reasonable weight
            average_weight = self._get_average_edge_weight()
            self.graph.add_edge(node, connected_node, average_weight)

            # Add to connected set
            connected.add(node)

        self.logger.info("Graph connectivity ensured.")
        return False

    def _get_average_edge_weight(self):
        """Calculate the average edge weight in the graph."""
        total = 0
        count = 0

        for source in self.graph.adjacency_list:
            for target, weight in self.graph.adjacency_list[source].items():
                total += weight
                count += 1

        return total / max(1, count)  # Avoid division by zero


def _name_similarity(str1, str2):
    """Simple string similarity metric based on common substrings."""
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0

    # Count common characters
    common = sum(1 for c1, c2 in zip(str1, str2) if c1 == c2)
    return common / max_len