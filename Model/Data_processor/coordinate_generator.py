# Model/data_processor/coordinate_generator.py
"""
Class for generating coordinates for hospital departments based on transport times.
Uses a force-directed layout algorithm to position nodes with increased spacing.
"""
import numpy as np
import logging
import random
from scipy.spatial.distance import pdist, squareform


class CoordinateGenerator:
    """
    Generates 2D coordinates for hospital departments based on their transport times.
    Uses a force-directed layout algorithm to position nodes in a way that approximates
    the relative distances between them.
    """

    def __init__(self, graph, canvas_width=1200, canvas_height=900):  # Increased canvas size
        """
        Initialize the CoordinateGenerator.

        Args:
            graph: Graph instance with nodes and edges
            canvas_width: Width of the visualization canvas
            canvas_height: Height of the visualization canvas
        """
        self.graph = graph
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up a logger for the CoordinateGenerator."""
        logger = logging.getLogger("CoordinateGenerator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def generate_coordinates(self, iterations=2000, temperature=0.15, cooling_factor=0.99, repulsion_force=5.0):
        """
        Generate coordinates for all nodes using force-directed placement.
        This algorithm positions nodes so that connected nodes are closer together,
        with edge weights influencing the ideal distance.

        Args:
            iterations: Number of iterations to run the force-directed algorithm
            temperature: Initial "temperature" controlling how much nodes can move
            cooling_factor: Factor to reduce temperature each iteration
            repulsion_force: Strength of repulsion between nodes (increased)

        Returns:
            bool: True if coordinates were generated successfully
        """
        self.logger.info("Generating coordinates using force-directed placement with increased spacing...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot generate coordinates.")
            return False

        n_nodes = len(nodes)

        # Create a node_index mapping for easier reference
        node_index = {node: i for i, node in enumerate(nodes)}

        # Initialize random positions
        positions = np.random.rand(n_nodes, 2)  # Random 2D coordinates
        positions[:, 0] *= self.canvas_width * 0.8  # Scale to canvas
        positions[:, 1] *= self.canvas_height * 0.8

        # Center the positions
        positions[:, 0] += self.canvas_width * 0.1
        positions[:, 1] += self.canvas_height * 0.1

        # Create distance matrix based on edge weights
        ideal_distances = np.zeros((n_nodes, n_nodes))

        # Set ideal distances based on edge weights
        # Increased distance scale factor from 5 to 15 to create more space
        distance_scale_factor = 15.0

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    continue

                # Try to get edge weight in both directions
                weight_ij = self.graph.get_edge_weight(node_i, node_j)
                weight_ji = self.graph.get_edge_weight(node_j, node_i)

                if weight_ij is not None:
                    # sqrt of time to make layout more compact, but with increased scale factor
                    ideal_distances[i, j] = np.sqrt(weight_ij) * distance_scale_factor
                elif weight_ji is not None:
                    ideal_distances[i, j] = np.sqrt(weight_ji) * distance_scale_factor
                else:
                    # If nodes not directly connected, use a larger default distance
                    ideal_distances[i, j] = np.sqrt(100) * distance_scale_factor

        # Set a minimum ideal distance to prevent nodes from getting too close
        min_distance = 100  # Minimum distance between any two nodes
        ideal_distances = np.maximum(ideal_distances, min_distance)

        # Run force-directed placement
        temp = temperature
        for iteration in range(iterations):
            # Calculate forces
            forces = np.zeros((n_nodes, 2))

            # Calculate repulsive forces between all pairs of nodes
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue

                    # Vector from j to i
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]

                    # Distance between nodes
                    distance = max(0.1, np.sqrt(dx ** 2 + dy ** 2))

                    # Repulsive force - strengthened by repulsion_force parameter
                    k = repulsion_force  # Repulsion strength (increased from original)
                    f_rep = k / (distance ** 0.8)  # Changed exponent to increase repulsion at medium distances

                    # Normalize direction
                    dx /= distance
                    dy /= distance

                    # Add repulsive force
                    forces[i, 0] += dx * f_rep
                    forces[i, 1] += dy * f_rep

            # Calculate attractive forces for connected nodes
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i == j:
                        continue

                    # Check if nodes are connected
                    weight_ij = self.graph.get_edge_weight(node_i, node_j)
                    weight_ji = self.graph.get_edge_weight(node_j, node_i)

                    if weight_ij is not None or weight_ji is not None:
                        # Vector from i to j
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]

                        # Distance between nodes
                        distance = max(0.1, np.sqrt(dx ** 2 + dy ** 2))

                        # Ideal distance based on edge weight
                        ideal = ideal_distances[i, j]

                        # Weaker spring force to allow more separation
                        f_spring = (distance - ideal) / max(0.0001,
                                                            ideal) * 0.7  # Reduced spring force by multiplying by 0.7

                        # Normalize direction
                        dx /= distance
                        dy /= distance

                        # Add attractive force
                        forces[i, 0] += dx * f_spring
                        forces[i, 1] += dy * f_spring

            # Update positions based on forces
            for i in range(n_nodes):
                # Apply temperature to limit movement
                dx = forces[i, 0] * temp
                dy = forces[i, 1] * temp

                # Limit maximum movement per iteration
                max_move = 30  # Increased from 20 to allow larger movements
                dx = max(-max_move, min(max_move, dx))
                dy = max(-max_move, min(max_move, dy))

                # Update position
                positions[i, 0] += dx
                positions[i, 1] += dy

                # Keep within canvas bounds
                positions[i, 0] = max(0, min(self.canvas_width, positions[i, 0]))
                positions[i, 1] = max(0, min(self.canvas_height, positions[i, 1]))

            # Cool down temperature
            temp *= cooling_factor

            # Log progress
            if iteration % 100 == 0:
                self.logger.debug(f"Force-directed placement: iteration {iteration}/{iterations}")

        # Update graph with coordinates
        for i, node in enumerate(nodes):
            x, y = positions[i]
            self.graph.set_node_coordinates(node, x, y)

        # Scale the layout to ensure it uses the full canvas - using a smaller padding
        # to push nodes closer to edges and maximize spacing
        self.scale_layout(padding=30)

        self.logger.info("Coordinate generation complete with increased spacing.")
        return True

    def generate_simple_coordinates(self):
        """
        Generate simple coordinates based on a circle layout with increased radius.
        Useful when force-directed placement is too complex or doesn't converge well.

        Returns:
            bool: True if coordinates were generated successfully
        """
        self.logger.info("Generating simple circle layout coordinates with increased spacing...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot generate coordinates.")
            return False

        n_nodes = len(nodes)

        # Calculate center of canvas
        center_x = self.canvas_width / 2
        center_y = self.canvas_height / 2

        # Calculate radius - increased to 0.45 of canvas size
        radius = min(self.canvas_width, self.canvas_height) * 0.45

        # Assign positions in a circle
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n_nodes
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)

            self.graph.set_node_coordinates(node, x, y)

        self.logger.info("Simple coordinate generation complete with increased spacing.")
        return True

    def generate_grid_layout(self, spacing=180):  # Increased spacing from 120 to 180
        """
        Place departments in a grid layout with increased spacing.

        Args:
            spacing: Space between departments in pixels

        Returns:
            bool: True if layout was generated successfully
        """
        self.logger.info("Generating grid layout with increased spacing...")

        # Get all nodes
        departments = list(self.graph.adjacency_list.keys())
        if not departments:
            self.logger.error("No nodes in graph. Cannot generate grid layout.")
            return False

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(departments))))

        # Place nodes in a grid
        for i, dept in enumerate(departments):
            row = i // grid_size
            col = i % grid_size

            x = 100 + col * spacing
            y = 100 + row * spacing

            # Add some small random offset to avoid perfect alignment
            x += random.uniform(-15, 15)  # Slightly increased randomness
            y += random.uniform(-15, 15)

            self.graph.set_node_coordinates(dept, x, y)

        # Scale to fit the canvas
        self.scale_layout()

        self.logger.info(f"Generated grid layout for {len(departments)} departments with increased spacing")
        return True

    def scale_layout(self, padding=50):
        """
        Scale the layout to fit the canvas with proper padding.

        Args:
            padding: Padding in pixels around the edge of the canvas
        """
        # Get all coordinates
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            return

        # Find current bounds
        coords = [self.graph.get_node_coordinates(n) for n in nodes]
        x_values = [c[0] for c in coords]
        y_values = [c[1] for c in coords]

        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)

        # Calculate scale factors
        available_width = self.canvas_width - 2 * padding
        available_height = self.canvas_height - 2 * padding

        current_width = max(1, max_x - min_x)  # Avoid division by zero
        current_height = max(1, max_y - min_y)

        scale_x = available_width / current_width
        scale_y = available_height / current_height

        # Use the smaller scale to preserve aspect ratio
        scale = min(scale_x, scale_y)

        # Scale and center all nodes
        for node in nodes:
            x, y = self.graph.get_node_coordinates(node)

            # Scale and shift
            new_x = padding + (x - min_x) * scale
            new_y = padding + (y - min_y) * scale

            self.graph.set_node_coordinates(node, new_x, new_y)

        self.logger.info(f"Scaled layout by factor {scale:.2f}")

    def adjust_coordinates_by_department_type(self, department_types=None):
        """
        Adjust coordinates to group departments of similar types.
        This is a heuristic approach to make the layout more realistic.

        Args:
            department_types: Dictionary mapping department names to type categories.
                             If None, will try to infer from department names.

        Returns:
            bool: True if coordinates were adjusted successfully
        """
        self.logger.info("Adjusting coordinates by department type with increased group separation...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot adjust coordinates.")
            return False

        # If department types not provided, try to infer from names
        if department_types is None:
            department_types = self._infer_department_types(nodes)

        # Group nodes by type
        type_groups = {}
        for node, dept_type in department_types.items():
            if dept_type not in type_groups:
                type_groups[dept_type] = []
            type_groups[dept_type].append(node)

        # Calculate current centroid of the layout
        centroid_x = 0
        centroid_y = 0
        for node in nodes:
            x, y = self.graph.get_node_coordinates(node)
            centroid_x += x
            centroid_y += y
        centroid_x /= len(nodes)
        centroid_y /= len(nodes)

        # Assign "regions" to each department type
        # We divide the canvas into sections for different department types
        num_types = len(type_groups)
        type_regions = {}

        # Simple case: place different types in different regions
        if num_types <= 4:
            # Four quadrants - pushed further to corners
            regions = [
                (self.canvas_width * 0.2, self.canvas_height * 0.2),  # Top left
                (self.canvas_width * 0.8, self.canvas_height * 0.2),  # Top right
                (self.canvas_width * 0.2, self.canvas_height * 0.8),  # Bottom left
                (self.canvas_width * 0.8, self.canvas_height * 0.8),  # Bottom right
            ]

            for i, dept_type in enumerate(type_groups.keys()):
                type_regions[dept_type] = regions[i % len(regions)]
        else:
            # More complex: arrange in a circle with larger radius
            radius = min(self.canvas_width, self.canvas_height) * 0.4  # Increased from 0.3
            for i, dept_type in enumerate(type_groups.keys()):
                angle = 2 * np.pi * i / num_types
                x = self.canvas_width / 2 + radius * np.cos(angle)
                y = self.canvas_height / 2 + radius * np.sin(angle)
                type_regions[dept_type] = (x, y)

        # Move departments toward their type's region
        adjustment_factor = 0.4  # Increased from 0.3 for stronger grouping
        for dept_type, region_center in type_regions.items():
            for node in type_groups.get(dept_type, []):
                x, y = self.graph.get_node_coordinates(node)

                # Calculate vector toward region center
                dx = region_center[0] - x
                dy = region_center[1] - y

                # Apply adjustment
                new_x = x + dx * adjustment_factor
                new_y = y + dy * adjustment_factor

                # Update coordinates
                self.graph.set_node_coordinates(node, new_x, new_y)

        self.logger.info("Coordinate adjustment by department type complete with increased separation.")
        return True

    def _infer_department_types(self, nodes):
        """
        Attempt to infer department types from their names.
        This is a heuristic approach using common keywords.

        Args:
            nodes: List of department names

        Returns:
            dict: Dictionary mapping department names to inferred types
        """
        # Define common keywords for different department types
        type_keywords = {
            'Emergency': ['emergency', 'er', 'trauma', 'acute'],
            'Surgery': ['surgery', 'surgical', 'operating', 'operation', 'or'],
            'Inpatient': ['ward', 'inpatient', 'patient', 'bed', 'room', 'icu', 'intensive'],
            'Diagnostic': ['radiology', 'imaging', 'xray', 'x-ray', 'mri', 'ct', 'scan', 'diagnostic', 'lab',
                           'laboratory', 'pathology'],
            'Outpatient': ['clinic', 'outpatient', 'consultation', 'therapy'],
            'Support': ['pharmacy', 'supply', 'storage', 'kitchen', 'cafeteria', 'admin', 'office', 'reception',
                        'entrance', 'lounge']
        }

        # Default type
        default_type = 'Other'

        # Map nodes to types
        node_types = {}
        for node in nodes:
            node_lower = node.lower()

            for dept_type, keywords in type_keywords.items():
                if any(keyword in node_lower for keyword in keywords):
                    node_types[node] = dept_type
                    break
            else:
                node_types[node] = default_type

        return node_types

    def apply_jitter(self, amount=45):  # Increased from 30 to 45
        """
        Apply small random offsets to coordinates to avoid perfect overlaps.

        Args:
            amount: Maximum jitter amount in pixels

        Returns:
            bool: True if jitter was applied successfully
        """
        self.logger.info(f"Applying coordinate jitter (max {amount} pixels)...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot apply jitter.")
            return False

        # Apply jitter to each node
        for node in nodes:
            x, y = self.graph.get_node_coordinates(node)

            # Add small random offsets
            x += random.uniform(-amount, amount)
            y += random.uniform(-amount, amount)

            # Keep within canvas bounds
            x = max(0, min(self.canvas_width, x))
            y = max(0, min(self.canvas_height, y))

            # Update coordinates
            self.graph.set_node_coordinates(node, x, y)

        self.logger.info("Coordinate jitter applied.")
        return True

    def ensure_minimum_distance(self, min_distance=80):
        """
        Ensure all nodes are at least a minimum distance apart.

        Args:
            min_distance: Minimum acceptable distance between nodes

        Returns:
            bool: True if adjustments were made successfully
        """
        self.logger.info(f"Ensuring minimum distance of {min_distance} pixels between nodes...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot ensure minimum distance.")
            return False

        n_nodes = len(nodes)
        if n_nodes <= 1:
            return True

        # Get current positions
        positions = np.array([self.graph.get_node_coordinates(node) for node in nodes])

        # Flag to track if we needed to make adjustments
        adjustments_made = False

        # Maximum number of iterations to prevent infinite loops
        max_iterations = 50

        for iteration in range(max_iterations):
            # Calculate pairwise distances
            distances = squareform(pdist(positions))

            # Set diagonal to a large value to ignore self-distances
            np.fill_diagonal(distances, float('inf'))

            # Find pairs of nodes that are too close
            too_close = distances < min_distance

            if not np.any(too_close):
                # All nodes are far enough apart
                break

            # Apply small repulsive forces to separate nodes
            adjustments_made = True
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if distances[i, j] < min_distance:
                        # Vector from j to i
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]

                        # Current distance
                        distance = distances[i, j]

                        # Desired separation
                        sep = min_distance - distance

                        # Normalize and apply separation
                        if distance > 0:  # Avoid division by zero
                            dx = dx / distance * sep * 0.5
                            dy = dy / distance * sep * 0.5

                            # Move both nodes apart
                            positions[i, 0] += dx
                            positions[i, 1] += dy
                            positions[j, 0] -= dx
                            positions[j, 1] -= dy
                        else:
                            # If nodes are at exactly the same position, move randomly
                            positions[i, 0] += random.uniform(5, 15)
                            positions[i, 1] += random.uniform(5, 15)
                            positions[j, 0] -= random.uniform(5, 15)
                            positions[j, 1] -= random.uniform(5, 15)

            # Keep within canvas bounds
            positions[:, 0] = np.clip(positions[:, 0], 0, self.canvas_width)
            positions[:, 1] = np.clip(positions[:, 1], 0, self.canvas_height)

        # Update graph with adjusted coordinates
        for i, node in enumerate(nodes):
            self.graph.set_node_coordinates(node, positions[i, 0], positions[i, 1])

        if adjustments_made:
            self.logger.info(f"Made adjustments to ensure minimum distance in {iteration + 1} iterations.")
        else:
            self.logger.info("No adjustments needed, nodes already satisfy minimum distance.")

        return True

    def spread_nodes(self, expansion_factor=1.2):
        """
        Spread out all nodes by scaling coordinates from the center.

        Args:
            expansion_factor: Factor by which to expand the layout

        Returns:
            bool: True if spreading was successful
        """
        self.logger.info(f"Spreading nodes by factor {expansion_factor}...")

        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot spread nodes.")
            return False

        # Calculate center of layout
        coords = [self.graph.get_node_coordinates(n) for n in nodes]
        x_values = [c[0] for c in coords]
        y_values = [c[1] for c in coords]

        center_x = sum(x_values) / len(x_values)
        center_y = sum(y_values) / len(y_values)

        # Expand from center
        for node in nodes:
            x, y = self.graph.get_node_coordinates(node)

            # Vector from center
            dx = x - center_x
            dy = y - center_y

            # Scale vector
            new_x = center_x + dx * expansion_factor
            new_y = center_y + dy * expansion_factor

            # Keep within canvas bounds
            new_x = max(0, min(self.canvas_width, new_x))
            new_y = max(0, min(self.canvas_height, new_y))

            self.graph.set_node_coordinates(node, new_x, new_y)

        self.logger.info(f"Spread nodes by factor {expansion_factor}")
        return True

    def export_coordinates(self, output_file=None):
        """
        Export node coordinates to a file.

        Args:
            output_file: Path to output file. If None, returns coordinate dict.

        Returns:
            dict or bool: Dictionary of coordinates if output_file is None, else True on success
        """
        # Get all nodes
        nodes = list(self.graph.adjacency_list.keys())
        if not nodes:
            self.logger.error("No nodes in graph. Cannot export coordinates.")
            return False if output_file else {}

        # Collect coordinates
        coordinates = {}
        for node in nodes:
            coordinates[node] = self.graph.get_node_coordinates(node)

        # Return or write to file
        if output_file is None:
            return coordinates

        try:
            import json
            with open(output_file, 'w') as f:
                json.dump(coordinates, f, indent=2)
            self.logger.info(f"Coordinates exported to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting coordinates: {str(e)}")
            return False